import asyncio
import csv
import os
import random
import sys
from datetime import datetime
from typing import List

from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI
from transformers import AutoTokenizer

logger = init_logger(__name__)


def _calc_stats(values: List[float]) -> tuple[float, float, float, float]:
    values = sorted(values)
    n = len(values)
    return (
        sum(values) / n,
        values[min(int(n * 0.95), n - 1)],
        values[min(int(n * 0.99), n - 1)],
        values[-1],
    )


def _emit_benchmark_summary(results, output_csv: str) -> None:
    all_tics = [r.tics for r in results]
    first_times = []
    token_times = []
    e2e_times = []
    for tics in all_tics:
        deltas = [tics[i + 1] - tics[i] for i in range(len(tics) - 1)]
        first_times.append(deltas[0])
        token_times.extend(deltas[1:])
        e2e_times.append(tics[-1] - tics[0])

    ttft = _calc_stats(first_times)
    tpot = _calc_stats(token_times)
    e2e = _calc_stats(e2e_times)

    min_time = min(min(x) for x in all_tics)
    max_time = max(max(x) for x in all_tics)
    duration = max_time - min_time
    num_tokens = sum(len(x) for x in all_tics)
    num_requests = len(all_tics)
    throughput_token = num_tokens / duration
    throughput_req = num_requests / duration

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "num_requests",
                    "num_tokens",
                    "duration_s",
                    "ttft_avg_ms",
                    "ttft_p95_ms",
                    "ttft_p99_ms",
                    "ttft_max_ms",
                    "tpot_avg_ms",
                    "tpot_p95_ms",
                    "tpot_p99_ms",
                    "tpot_max_ms",
                    "e2e_avg_s",
                    "e2e_p95_s",
                    "e2e_p99_s",
                    "e2e_max_s",
                    "throughput_token_s",
                    "throughput_req_s",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                num_requests,
                num_tokens,
                duration,
                ttft[0] * 1000,
                ttft[1] * 1000,
                ttft[2] * 1000,
                ttft[3] * 1000,
                tpot[0] * 1000,
                tpot[1] * 1000,
                tpot[2] * 1000,
                tpot[3] * 1000,
                e2e[0],
                e2e[1],
                e2e[2],
                e2e[3],
                throughput_token,
                throughput_req,
            ]
        )

    logger.info("P0 benchmark summary csv saved: %s", output_csv)
    logger.info(
        "P0 summary: req=%d tokens=%d duration=%.2fs TTFT(avg/p95/p99/max)=%.2f/%.2f/%.2f/%.2fms TPOT(avg/p95/p99/max)=%.2f/%.2f/%.2f/%.2fms E2E(avg/p95/p99/max)=%.2f/%.2f/%.2f/%.2fs throughput(token/s,req/s)=%.2f,%.4f",
        num_requests,
        num_tokens,
        duration,
        ttft[0] * 1000,
        ttft[1] * 1000,
        ttft[2] * 1000,
        ttft[3] * 1000,
        tpot[0] * 1000,
        tpot[1] * 1000,
        tpot[2] * 1000,
        tpot[3] * 1000,
        e2e[0],
        e2e[1],
        e2e[2],
        e2e[3],
        throughput_token,
        throughput_req,
    )


async def main():
    try:
        random.seed(42)  # reproducibility

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            result = []
            for _ in range(max_bs):
                length = random.randint(1, MAX_INPUT)
                message = generate_prompt(tokenizer, length)
                result.append(message)
                await asyncio.sleep(0)
            return result

        TEST_BS = [64]
        PORT = 1919
        MAX_INPUT = 8192
        # Create the async client
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="") as client:
            MODEL = await get_model_name(client)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            logger.info(f"Loaded tokenizer from {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            try:
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                raise e from e

            msgs = await gen_task
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
            logger.info(f"Generated {len(msgs)} test messages")

            logger.info("Running benchmark...")
            summary_csv = os.getenv("MINISGL_BENCH_SUMMARY_CSV", "benchmark/online/bench_summary.csv")
            for batch_size in TEST_BS:
                try:
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL
                    )
                    process_benchmark_results(results)
                    _emit_benchmark_summary(results, summary_csv)
                except Exception as e:
                    logger.info(f"Error with batch size {batch_size}: {e}")
                    continue
            logger.info("Benchmark completed.")

    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
