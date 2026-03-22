from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from contextlib import closing
from urllib import request

import pytest
import torch


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_get_json(url: str, timeout: float = 2.0):
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_until_ready(base_url: str, timeout_sec: float = 900.0) -> None:
    deadline = time.time() + timeout_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            data = _http_get_json(f"{base_url}/v1/models", timeout=3.0)
            assert "data" in data
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2.0)
    raise AssertionError(f"Server is not ready before timeout: {last_err}")


def _read_stream_until_done(base_url: str, model_path: str, timeout_sec: float = 180.0) -> str:
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": "Reply with one short sentence."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    deadline = time.time() + timeout_sec
    with request.urlopen(req, timeout=timeout_sec) as resp:
        chunks: list[str] = []
        while time.time() < deadline:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            chunks.append(text)
            if text.endswith("[DONE]"):
                return "\n".join(chunks)

    raise AssertionError("Did not receive [DONE] from streaming response")


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need at least 2 GPUs for EP smoke test")
def test_ep_server_smoke_gpu():
    model_path = os.environ.get("MINISGL_EP_MODEL", "").strip()
    if not model_path:
        pytest.skip("Set MINISGL_EP_MODEL to a MoE model path/repo to run this smoke test")

    if os.environ.get("MINISGL_RUN_GPU_SMOKE", "0") != "1":
        pytest.skip("Set MINISGL_RUN_GPU_SMOKE=1 to enable GPU smoke test")

    world_size = int(os.environ.get("MINISGL_EP_WORLD_SIZE", "2"))
    if world_size < 2:
        pytest.skip("MINISGL_EP_WORLD_SIZE must be >= 2")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"Need at least {world_size} GPUs, but found {torch.cuda.device_count()}"
        )

    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    cmd = [
        "python",
        "-m",
        "minisgl",
        "--model",
        model_path,
        "--tp",
        str(world_size),
        "--ep-size",
        str(world_size),
        "--moe-backend",
        "ep",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    env = os.environ.copy()
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
        env=env,
    )

    try:
        _wait_until_ready(base_url)
        stream_dump = _read_stream_until_done(base_url, model_path)
        assert "[DONE]" in stream_dump
    except Exception:  # noqa: BLE001
        logs = ""
        if proc.stdout is not None:
            try:
                logs = proc.stdout.read()
            except Exception:  # noqa: BLE001
                logs = "<failed to read server logs>"
        raise AssertionError(f"EP smoke test failed. Server logs:\n{logs}")
    finally:
        _terminate_process_group(proc)
