from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    MetricsReportMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


@dataclass
class MetricsState:
    """State for metrics tracking in API server."""
    timestamp: float = 0.0
    running_requests: int = 0
    queued_requests: int = 0
    completed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    throughput_tokens_per_sec: float = 0.0
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p90_ttft: float = 0.0
    p99_ttft: float = 0.0
    # KV cache metrics (SGLang compatible)
    num_used_tokens: int = 0
    max_total_num_tokens: int = 0
    token_usage: float = 0.0
    # Cache metrics (token-level)
    cache_hit_rate: float = 0.0
    cache_hit_tokens: int = 0
    cache_prefill_tokens: int = 0
    # Cache metrics (legacy, kept for compatibility)
    cache_hits: int = 0
    cache_misses: int = 0
    # Queue time metrics
    avg_queue_time: float = 0.0
    p50_queue_time: float = 0.0
    p99_queue_time: float = 0.0

    def update_from_msg(self, msg: MetricsReportMsg) -> None:
        """Update state from MetricsReportMsg."""
        self.timestamp = time.time()
        self.running_requests = msg.running_requests
        self.queued_requests = msg.queued_requests
        self.completed_requests = msg.completed_requests
        self.total_input_tokens = msg.total_input_tokens
        self.total_output_tokens = msg.total_output_tokens
        self.throughput_tokens_per_sec = msg.throughput_tokens_per_sec
        self.avg_ttft = msg.avg_ttft
        self.p50_ttft = msg.p50_ttft
        self.p90_ttft = msg.p90_ttft
        self.p99_ttft = msg.p99_ttft
        # KV cache metrics
        self.num_used_tokens = msg.num_used_tokens
        self.max_total_num_tokens = msg.max_total_num_tokens
        self.token_usage = msg.token_usage
        # Cache metrics (token-level)
        self.cache_hit_rate = msg.cache_hit_rate
        self.cache_hit_tokens = msg.cache_hit_tokens
        self.cache_prefill_tokens = msg.cache_prefill_tokens
        # Cache metrics (legacy)
        self.cache_hits = msg.cache_hits
        self.cache_misses = msg.cache_misses
        # Queue time metrics
        self.avg_queue_time = msg.avg_queue_time
        self.p50_queue_time = msg.p50_queue_time
        self.p99_queue_time = msg.p99_queue_time

    def to_prometheus(self) -> str:
        """Convert metrics to Prometheus exposition format."""
        lines = [
            # Request counts
            "# HELP minisgl_running_requests Number of running requests",
            "# TYPE minisgl_running_requests gauge",
            f"minisgl_running_requests {self.running_requests}",
            "# HELP minisgl_queued_requests Number of queued requests",
            "# TYPE minisgl_queued_requests gauge",
            f"minisgl_queued_requests {self.queued_requests}",
            "# HELP minisgl_completed_requests Total number of completed requests",
            "# TYPE minisgl_completed_requests counter",
            f"minisgl_completed_requests {self.completed_requests}",
            # Token counts
            "# HELP minisgl_total_input_tokens Total number of input tokens processed",
            "# TYPE minisgl_total_input_tokens counter",
            f"minisgl_total_input_tokens {self.total_input_tokens}",
            "# HELP minisgl_total_output_tokens Total number of output tokens generated",
            "# TYPE minisgl_total_output_tokens counter",
            f"minisgl_total_output_tokens {self.total_output_tokens}",
            # Throughput
            "# HELP minisgl_output_throughput Output throughput in tokens per second",
            "# TYPE minisgl_output_throughput gauge",
            f"minisgl_output_throughput {self.throughput_tokens_per_sec:.2f}",
            # TTFT metrics
            "# HELP minisgl_ttft_avg Average time to first token in seconds",
            "# TYPE minisgl_ttft_avg gauge",
            f"minisgl_ttft_avg {self.avg_ttft:.6f}",
            "# HELP minisgl_ttft_p50 P50 time to first token in seconds",
            "# TYPE minisgl_ttft_p50 gauge",
            f"minisgl_ttft_p50 {self.p50_ttft:.6f}",
            "# HELP minisgl_ttft_p90 P90 time to first token in seconds",
            "# TYPE minisgl_ttft_p90 gauge",
            f"minisgl_ttft_p90 {self.p90_ttft:.6f}",
            "# HELP minisgl_ttft_p99 P99 time to first token in seconds",
            "# TYPE minisgl_ttft_p99 gauge",
            f"minisgl_ttft_p99 {self.p99_ttft:.6f}",
            # KV cache metrics (SGLang compatible)
            "# HELP minisgl_num_used_tokens Number of used tokens in KV cache",
            "# TYPE minisgl_num_used_tokens gauge",
            f"minisgl_num_used_tokens {self.num_used_tokens}",
            "# HELP minisgl_max_total_num_tokens Maximum total number of tokens in KV cache pool",
            "# TYPE minisgl_max_total_num_tokens gauge",
            f"minisgl_max_total_num_tokens {self.max_total_num_tokens}",
            "# HELP minisgl_token_usage The token usage ratio (used/total)",
            "# TYPE minisgl_token_usage gauge",
            f"minisgl_token_usage {self.token_usage:.6f}",
            # Cache metrics (token-level)
            "# HELP minisgl_cache_hit_rate The prefix cache hit rate (tokens)",
            "# TYPE minisgl_cache_hit_rate gauge",
            f"minisgl_cache_hit_rate {self.cache_hit_rate:.4f}",
            "# HELP minisgl_cache_hit_tokens Total number of cache hit tokens",
            "# TYPE minisgl_cache_hit_tokens counter",
            f"minisgl_cache_hit_tokens {self.cache_hit_tokens}",
            "# HELP minisgl_cache_prefill_tokens Total number of prefill tokens",
            "# TYPE minisgl_cache_prefill_tokens counter",
            f"minisgl_cache_prefill_tokens {self.cache_prefill_tokens}",
            # Cache metrics (legacy)
            "# HELP minisgl_cache_hits Total number of cache hits (requests)",
            "# TYPE minisgl_cache_hits counter",
            f"minisgl_cache_hits {self.cache_hits}",
            "# HELP minisgl_cache_misses Total number of cache misses (requests)",
            "# TYPE minisgl_cache_misses counter",
            f"minisgl_cache_misses {self.cache_misses}",
            # Queue time metrics
            "# HELP minisgl_queue_time_avg Average queue time in seconds",
            "# TYPE minisgl_queue_time_avg gauge",
            f"minisgl_queue_time_avg {self.avg_queue_time:.6f}",
            "# HELP minisgl_queue_time_p50 P50 queue time in seconds",
            "# TYPE minisgl_queue_time_p50 gauge",
            f"minisgl_queue_time_p50 {self.p50_queue_time:.6f}",
            "# HELP minisgl_queue_time_p99 P99 queue time in seconds",
            "# TYPE minisgl_queue_time_p99 gauge",
            f"minisgl_queue_time_p99 {self.p99_queue_time:.6f}",
        ]
        return "\n".join(lines) + "\n"


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # Metrics state
    metrics: MetricsState = field(default_factory=lambda: MetricsState())

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    def update_metrics(self, msg: MetricsReportMsg) -> None:
        """Update metrics from scheduler."""
        self.metrics.update_from_msg(msg)

    def get_metrics_prometheus(self) -> str:
        """Get metrics in Prometheus format."""
        return self.metrics.to_prometheus()

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            if isinstance(msg, MetricsReportMsg):
                self.update_metrics(msg)
                continue
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


@app.get("/metrics")
async def metrics():
    """Expose metrics in Prometheus format."""
    state = get_global_state()
    return PlainTextResponse(state.get_metrics_prometheus(), media_type="text/plain")


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


async def read_stdin():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        line = line.decode().rstrip("\n")


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            need_stop = False
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                if need_stop:
                    break
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
