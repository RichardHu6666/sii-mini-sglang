from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List
from urllib import request

from minisgl.benchmark.client import benchmark_one, generate_prompt, get_model_name
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Scenario:
    name: str
    port: int
    ep_size: int
    moe_backend: str


@dataclass(frozen=True)
class BenchmarkMetrics:
    scenario: str
    requests: int
    tokens: int
    duration_s: float
    ttft_ms_avg: float
    ttft_ms_p50: float
    ttft_ms_p90: float
    tpot_ms_avg: float
    tpot_ms_p50: float
    tpot_ms_p90: float
    e2e_s_avg: float
    e2e_s_p50: float
    e2e_s_p90: float
    tok_per_s: float
    req_per_s: float


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_until_ready(base_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            data = _http_get_json(f"{base_url}/v1/models", timeout=3.0)
            if "data" in data:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0)
    raise RuntimeError(f"Server is not ready before timeout, last error: {last_err}")


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def _percentile(sorted_values: List[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(int(len(sorted_values) * ratio), len(sorted_values) - 1)
    return sorted_values[idx]


def _compute_metrics(scenario: str, results: list[list[float]]) -> BenchmarkMetrics:
    first_times: list[float] = []
    per_token_times: list[float] = []
    e2e_times: list[float] = []
    num_tokens = 0

    for tics in results:
        deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
        if not deltas:
            continue
        first_times.append(deltas[0])
        per_token_times.extend(deltas[1:])
        e2e_times.append(tics[-1] - tics[0])
        num_tokens += len(tics)

    if not first_times or not per_token_times or not e2e_times:
        raise RuntimeError("Empty benchmark results, cannot compute metrics")

    first_times.sort()
    per_token_times.sort()
    e2e_times.sort()

    min_time = min(min(t) for t in results)
    max_time = max(max(t) for t in results)
    duration = max_time - min_time
    if duration <= 0:
        raise RuntimeError("Invalid duration from benchmark results")

    return BenchmarkMetrics(
        scenario=scenario,
        requests=len(results),
        tokens=num_tokens,
        duration_s=duration,
        ttft_ms_avg=1000.0 * sum(first_times) / len(first_times),
        ttft_ms_p50=1000.0 * _percentile(first_times, 0.5),
        ttft_ms_p90=1000.0 * _percentile(first_times, 0.9),
        tpot_ms_avg=1000.0 * sum(per_token_times) / len(per_token_times),
        tpot_ms_p50=1000.0 * _percentile(per_token_times, 0.5),
        tpot_ms_p90=1000.0 * _percentile(per_token_times, 0.9),
        e2e_s_avg=sum(e2e_times) / len(e2e_times),
        e2e_s_p50=_percentile(e2e_times, 0.5),
        e2e_s_p90=_percentile(e2e_times, 0.9),
        tok_per_s=num_tokens / duration,
        req_per_s=len(results) / duration,
    )


def _render_markdown_table(base: BenchmarkMetrics, exp: BenchmarkMetrics) -> str:
    def pct(new: float, old: float) -> str:
        if old == 0:
            return "n/a"
        return f"{(new - old) / old * 100.0:+.1f}%"

    lines = [
        "| Metric | Non-EP (fused) | EP | Delta (EP vs Non-EP) |",
        "|---|---:|---:|---:|",
        f"| Throughput (token/s) | {base.tok_per_s:.2f} | {exp.tok_per_s:.2f} | {pct(exp.tok_per_s, base.tok_per_s)} |",
        f"| Throughput (req/s) | {base.req_per_s:.4f} | {exp.req_per_s:.4f} | {pct(exp.req_per_s, base.req_per_s)} |",
        f"| TTFT avg (ms) | {base.ttft_ms_avg:.2f} | {exp.ttft_ms_avg:.2f} | {pct(exp.ttft_ms_avg, base.ttft_ms_avg)} |",
        f"| TTFT p50 (ms) | {base.ttft_ms_p50:.2f} | {exp.ttft_ms_p50:.2f} | {pct(exp.ttft_ms_p50, base.ttft_ms_p50)} |",
        f"| TPOT avg (ms) | {base.tpot_ms_avg:.2f} | {exp.tpot_ms_avg:.2f} | {pct(exp.tpot_ms_avg, base.tpot_ms_avg)} |",
        f"| TPOT p50 (ms) | {base.tpot_ms_p50:.2f} | {exp.tpot_ms_p50:.2f} | {pct(exp.tpot_ms_p50, base.tpot_ms_p50)} |",
        f"| E2E avg (s) | {base.e2e_s_avg:.4f} | {exp.e2e_s_avg:.4f} | {pct(exp.e2e_s_avg, base.e2e_s_avg)} |",
        f"| Duration (s) | {base.duration_s:.2f} | {exp.duration_s:.2f} | {pct(exp.duration_s, base.duration_s)} |",
    ]
    return "\n".join(lines)


async def _run_load(
    base_url: str,
    model_name: str,
    prompts: list[str],
    output_lens: list[int],
    concurrency: int,
) -> list[list[float]]:
    async with OpenAI(base_url=f"{base_url}/v1", api_key="") as client:
        sem = asyncio.Semaphore(concurrency)

        async def one(prompt: str, out_len: int) -> list[float]:
            async with sem:
                result = await benchmark_one(
                    client=client,
                    prompt=prompt,
                    output_length=out_len,
                    model=model_name,
                    pbar=False,
                    extra_body={"ignore_eos": True, "top_k": 1},
                )
                return result.tics

        tasks = [one(p, o) for p, o in zip(prompts, output_lens, strict=True)]
        return await asyncio.gather(*tasks)


async def _prepare_prompts(model_name: str, num_prompts: int, input_len: int, output_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [generate_prompt(tokenizer, input_len) for _ in range(num_prompts)]
    output_lens = [output_len] * num_prompts
    return prompts, output_lens


def _start_server(
    model_path: str,
    scenario: Scenario,
    tp: int,
    host: str,
    log_file: Path,
) -> subprocess.Popen[str]:
    cmd = [
        "python",
        "-m",
        "minisgl",
        "--model",
        model_path,
        "--tp",
        str(tp),
        "--ep-size",
        str(scenario.ep_size),
        "--moe-backend",
        scenario.moe_backend,
        "--host",
        host,
        "--port",
        str(scenario.port),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fp = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=fp,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=os.environ.copy(),
    )
    return proc


async def _run_scenario(
    model_path: str,
    scenario: Scenario,
    tp: int,
    host: str,
    ready_timeout_s: float,
    prompts: list[str],
    output_lens: list[int],
    concurrency: int,
    log_dir: Path,
) -> BenchmarkMetrics:
    base_url = f"http://{host}:{scenario.port}"
    server_log = log_dir / f"server_{scenario.name}.log"
    proc = _start_server(model_path, scenario, tp, host, server_log)
    try:
        _wait_until_ready(base_url, timeout_sec=ready_timeout_s)
        async with OpenAI(base_url=f"{base_url}/v1", api_key="") as client:
            model_name = await get_model_name(client)
        raw = await _run_load(base_url, model_name, prompts, output_lens, concurrency)
        return _compute_metrics(scenario.name, raw)
    finally:
        _terminate_process_group(proc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EP vs non-EP A/B benchmark")
    parser.add_argument("--model", required=True, help="Model path or HF repo id")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--base-port", type=int, default=1919, help="Port for baseline fused")
    parser.add_argument("--exp-port", type=int, default=1920, help="Port for EP run")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--ready-timeout-s", type=float, default=1200.0)
    parser.add_argument(
        "--out-dir",
        default="benchmark/online/results",
        help="Directory to write markdown/json and server logs",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused = Scenario(name="fused", port=args.base_port, ep_size=1, moe_backend="fused")
    ep = Scenario(name="ep", port=args.exp_port, ep_size=args.tp, moe_backend="ep")

    prompts, output_lens = await _prepare_prompts(
        model_name=args.model,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
    )

    base_metrics = await _run_scenario(
        model_path=args.model,
        scenario=fused,
        tp=args.tp,
        host=args.host,
        ready_timeout_s=args.ready_timeout_s,
        prompts=prompts,
        output_lens=output_lens,
        concurrency=args.concurrency,
        log_dir=out_dir,
    )
    exp_metrics = await _run_scenario(
        model_path=args.model,
        scenario=ep,
        tp=args.tp,
        host=args.host,
        ready_timeout_s=args.ready_timeout_s,
        prompts=prompts,
        output_lens=output_lens,
        concurrency=args.concurrency,
        log_dir=out_dir,
    )

    markdown = _render_markdown_table(base_metrics, exp_metrics)
    ts = time.strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"ep_ab_{ts}.md"
    json_path = out_dir / f"ep_ab_{ts}.json"
    md_path.write_text(markdown + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "base": asdict(base_metrics),
                "exp": asdict(exp_metrics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(markdown)
    print(f"\nSaved markdown report: {md_path}")
    print(f"Saved json report: {json_path}")


if __name__ == "__main__":
    asyncio.run(_main())
