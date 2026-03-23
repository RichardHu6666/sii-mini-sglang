"""
Online benchmark script for Mini-SGLang with enhanced metrics.

Features:
- Command-line arguments for concurrency, input/output lengths
- Export TTFT, throughput metrics to JSON
- Sample /metrics endpoint and plot time-series charts
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from minisgl.benchmark.client import (
    BenchOneResult,
    BenchmarkResult,
    RawResult,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini-SGLang online benchmark")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=1919, help="Server port")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of concurrent requests")
    parser.add_argument("--num-reqs", type=int, default=100, help="Number of requests to run (default: 100)")
    parser.add_argument("--input-len", type=int, default=1024, help="Input sequence length (only used when --prompts=random)")
    parser.add_argument("--output-len", type=int, default=1024, help="Output sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--metrics-interval", type=float, default=0.5, help="Interval in seconds to sample /metrics")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip sampling /metrics endpoint")
    parser.add_argument("--model-path", type=str, default=None, help="Model path for tokenizer (auto-detected if not set)")
    parser.add_argument("--prompts", type=str, default="./prompts.jsonl", choices=["./prompts.jsonl", "random"], help="Prompts source: ./prompts.jsonl (default) or random")
    parser.add_argument("--prefix-ratio", type=float, default=0.0, help="Ratio of shared prefix in generated prompts (0.0-1.0). Only used when --prompts=random")
    return parser.parse_args()


class MetricsSampler:
    """Samples /metrics endpoint and stores time-series data."""

    def __init__(self, host: str, port: int, interval: float, output_dir: Path):
        self.url = f"http://{host}:{port}/metrics"
        self.interval = interval
        self.output_dir = output_dir
        self.samples: List[Dict[str, Any]] = []
        self.start_time = 0.0
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def _sample_once(self, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Sample the /metrics endpoint once."""
        try:
            async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return None
                text = await resp.text()
                return self._parse_prometheus(text)
        except Exception as e:
            logger.debug(f"Failed to sample metrics: {e}")
            return None

    def _parse_prometheus(self, text: str) -> Dict[str, Any]:
        """Parse Prometheus format metrics."""
        metrics = {}
        for line in text.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return metrics

    async def _sample_loop(self):
        """Continuous sampling loop."""
        async with aiohttp.ClientSession() as session:
            while self.running:
                sample = await self._sample_once(session)
                if sample:
                    sample["timestamp"] = time.time() - self.start_time
                    self.samples.append(sample)
                await asyncio.sleep(self.interval)

    async def start(self):
        """Start sampling."""
        self.running = True
        self.start_time = time.time()
        self.task = asyncio.create_task(self._sample_loop())
        logger.info(f"Started metrics sampling at {self.url}")

    async def stop(self):
        """Stop sampling."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped metrics sampling, collected {len(self.samples)} samples")

    def save(self, filename: str):
        """Save samples to JSON and generate plots."""
        if not self.samples:
            logger.warning("No metrics samples to save")
            return

        # Save raw JSON
        json_path = self.output_dir / filename
        with open(json_path, "w") as f:
            json.dump(self.samples, f, indent=2)
        logger.info(f"Saved metrics samples to {json_path}")

        # Generate plots
        self._plot_metrics(filename.replace(".json", ""))

    def _plot_metrics(self, prefix: str):
        """Generate time-series plots from samples."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation")
            return

        if len(self.samples) < 2:
            logger.warning("Not enough samples for plotting")
            return

        timestamps = [s["timestamp"] for s in self.samples]

        # Metric groupings for combined plots
        # Note: Metrics in the same group should have similar scales for meaningful visualization
        # running_requests from server is now active_requests (actually being processed)
        metric_groups = {
            "request_counts": [
                ("minisgl_running_requests", "Decoding", "gauge"),
                ("minisgl_queued_requests", "Queued", "gauge"),
            ],
            "kv_cache_tokens": [
                ("minisgl_num_used_tokens", "Used Tokens", "gauge"),
                ("minisgl_max_total_num_tokens", "Max Capacity", "gauge"),
            ],
            "throughput": [
                ("minisgl_output_throughput", "Throughput", "gauge"),
            ],
            "cache_hit_rate": [
                ("minisgl_cache_hit_rate", "Hit Rate", "gauge"),
            ],
        }

        # Color palette for different metric categories
        colors = {
            "request_counts": ["#2E86AB", "#E94F37"],  # blue for active, red for queued
            "kv_cache_tokens": ["#845EC2", "#D65DB1"],
            "throughput": ["#00C9A7"],
            "cache_hit_rate": ["#9B5DE5"],
        }

        # Pre-calculate computed metrics (not needed now, kept for future extensions)
        for s in self.samples:
            pass

        # Filter to groups that have data
        available_groups = {}
        for group_name, metrics in metric_groups.items():
            available_metrics = []
            for metric, display_name, metric_type in metrics:
                if metric in self.samples[0] or metric == "token_usage_raw":
                    available_metrics.append((metric, display_name, metric_type))
            if available_metrics:
                available_groups[group_name] = available_metrics

        if not available_groups:
            logger.warning("No recognized metrics found in data")
            return

        # Create figure with grouped subplots
        num_groups = len(available_groups)
        fig, axes = plt.subplots(num_groups, 1, figsize=(14, 5 * num_groups))
        if num_groups == 1:
            axes = [axes]

        # Group titles
        group_titles = {
            "request_counts": "Request Counts",
            "kv_cache_tokens": "KV Cache Token Counts",
            "throughput": "Throughput",
            "cache_hit_rate": "Cache Hit Rate",
        }

        # Plot each group
        for idx, (group_name, metrics) in enumerate(available_groups.items()):
            ax = axes[idx]
            group_colors = colors.get(group_name, ["#1f77b4", "#ff7f0e", "#2ca02c"])

            for metric_idx, (metric, display_name, metric_type) in enumerate(metrics):
                values = [s.get(metric, 0) for s in self.samples]
                color = group_colors[metric_idx % len(group_colors)]

                ax.plot(
                    timestamps, values,
                    linewidth=2,
                    color=color,
                    label=display_name,
                    marker='o' if len(self.samples) <= 20 else '',
                    markersize=4,
                    alpha=0.9
                )

            ax.set_title(group_titles.get(group_name, group_name), fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel("Time (seconds)", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
            ax.set_facecolor('#FAFAFA')

        fig.suptitle("Mini-SGLang Benchmark Metrics", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path = self.output_dir / f"{prefix}_timeseries.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved metrics plot to {plot_path}")


async def run_with_semaphore(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int],
    model: str,
    concurrency: int,
) -> List[RawResult]:
    """
    Run benchmark with dynamic concurrency control.
    Keeps at most `concurrency` requests running at any time.
    When one finishes, immediately start the next one.

    TTFT is measured from when the request is scheduled (claimed from the queue)
    to when the first token is received, including any queuing delay on the server.
    """
    import asyncio
    import time

    results: List[RawResult] = []
    next_index = 0  # Next prompt index to schedule
    total = len(prompts)
    lock = asyncio.Lock()  # Shared lock for task distribution
    completed = 0

    async def run_one(index: int, start_time: float) -> RawResult:
        """Run a single request."""
        nonlocal completed
        prompt = prompts[index]
        output_len = output_lengths[index]
        result = await benchmark_one(client, prompt, output_len, model, pbar=False)
        completed += 1
        if completed % 10 == 0 or completed == total:
            logger.info(f"Progress: {completed}/{total} requests completed")
        # Replace the first tic with our scheduled start time to include queuing delay
        return RawResult(
            input_len=result.input_len,
            output_len=result.output_len,
            message=result.message,
            tics=[start_time] + result.tics[1:],
        )

    async def worker():
        """Worker that processes tasks from the shared queue."""
        nonlocal next_index
        while True:
            # Claim the next task atomically
            async with lock:
                if next_index >= total:
                    return
                current_index = next_index
                next_index += 1
                # Record the time when this request is scheduled
                schedule_time = time.perf_counter()

            # Run the request (outside lock to allow concurrency)
            result = await run_one(current_index, schedule_time)
            results.append(result)

    # Start worker tasks - use concurrency number of workers
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)

    return results


async def main():
    args = parse_args()

    try:
        random.seed(args.seed)

        # Setup output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the async client
        async with OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="") as client:
            if args.model_path:
                MODEL = args.model_path
            else:
                MODEL = await get_model_name(client)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            logger.info(f"Loaded tokenizer from {MODEL}")
            logger.info("Testing connection to server...")

            # Test connection
            try:
                test_msg = generate_prompt(tokenizer, 100)
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                if len(test_result.tics) <= 2:
                    logger.info("Server connection test failed")
                    return
                logger.info("Server connection successful")
            except Exception as e:
                logger.warning("Server connection failed")
                logger.warning(f"Make sure the server is running on http://{args.host}:{args.port}")
                raise e

            # Prepare prompts
            concurrency = args.concurrency
            num_reqs = args.num_reqs
            input_len = args.input_len
            output_len = args.output_len
            prefix_ratio = args.prefix_ratio

            # Load prompts from file or generate randomly
            if args.prompts == "random":
                # Generate random prompts with optional shared prefix
                if prefix_ratio > 0:
                    # Generate prompts with shared prefix for testing prefix cache
                    shared_prefix_len = int(input_len * prefix_ratio)
                    unique_suffix_len = input_len - shared_prefix_len

                    # Generate shared prefix (same for all prompts)
                    shared_prefix = generate_prompt(tokenizer, shared_prefix_len)

                    logger.info(f"Generating prompts with {prefix_ratio*100:.0f}% shared prefix "
                               f"({shared_prefix_len} tokens prefix, {unique_suffix_len} tokens suffix)")

                    # Generate unique suffix for each request
                    messages = []
                    for _ in range(num_reqs):
                        suffix = generate_prompt(tokenizer, unique_suffix_len)
                        message = shared_prefix + suffix
                        messages.append(message)

                    logger.info(f"Generated {len(messages)} prompts with shared prefix")
                else:
                    # Generate completely random prompts
                    messages = []
                    for _ in range(num_reqs):
                        length = random.randint(max(1, input_len // 2), input_len)
                        message = generate_prompt(tokenizer, length)
                        messages.append(message)
                    logger.info(f"Generated {len(messages)} test messages")

                logger.info(f"Input length: ~{input_len} tokens")
                all_prompts = messages
            else:
                # Load from ./prompts.jsonl
                prompts_path = Path(args.prompts)
                if not prompts_path.exists():
                    logger.error(f"Prompts file not found: {prompts_path}")
                    return
                with open(prompts_path, "r") as f:
                    file_prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

                # Use first num_reqs prompts, or all if file has fewer
                if len(file_prompts) >= num_reqs:
                    all_prompts = file_prompts[:num_reqs]
                    logger.info(f"Loaded {num_reqs} prompts from {prompts_path} (first {num_reqs} of {len(file_prompts)})")
                else:
                    all_prompts = file_prompts
                    logger.warning(f"prompts.jsonl has only {len(file_prompts)} prompts, using all of them (requested {num_reqs})")

            # Generate output lengths
            total_prompts = len(all_prompts)
            all_output_lengths = [random.randint(max(1, output_len // 2), output_len) for _ in range(total_prompts)]

            logger.info(f"Output length: ~{output_len} tokens")
            logger.info(f"Total prompts to run: {total_prompts}")
            logger.info(f"Max concurrency: {concurrency}")

            # Start metrics sampling
            metrics_sampler = None
            if not args.skip_metrics:
                metrics_sampler = MetricsSampler(args.host, args.port, args.metrics_interval, output_dir)
                await metrics_sampler.start()

            logger.info("Running benchmark...")
            try:
                # Run with dynamic concurrency control
                results = await run_with_semaphore(
                    client, all_prompts, all_output_lengths, MODEL, concurrency
                )

                # Process and display results
                logger.info("\n" + "=" * 60)
                logger.info("Benchmark Results Summary")
                logger.info("=" * 60)
                benchmark_result = process_benchmark_results(results, tokenizer)

                # Calculate and save metrics as JSON
                metrics_data = {
                    "config": {
                        "concurrency": concurrency,
                        "input_length": input_len,
                        "output_length": output_len,
                        "seed": args.seed,
                        "model": MODEL,
                        "timestamp": timestamp,
                        "prompts_source": args.prompts,
                        "prefix_ratio": prefix_ratio if args.prompts == "random" else None,
                        "total_requests": len(results),
                    },
                    "results": calculate_metrics(results),
                }

                json_path = output_dir / f"{timestamp}_metrics.json"
                with open(json_path, "w") as f:
                    json.dump(metrics_data, f, indent=2)
                logger.info(f"\nSaved metrics to {json_path}")

            finally:
                # Stop metrics sampling
                if metrics_sampler:
                    await metrics_sampler.stop()
                    metrics_sampler.save(f"{timestamp}_metrics_samples.json")

            logger.info("\nBenchmark completed.")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def calculate_metrics(results: List[Any]) -> Dict[str, Any]:
    """Calculate detailed metrics from benchmark results."""
    ttft_values = []
    tpot_values = []
    e2e_values = []
    total_tokens = 0

    for r in results:
        tics = r.tics
        if len(tics) < 2:
            continue

        # TTFT: time to first token
        ttft = tics[1] - tics[0]
        ttft_values.append(ttft)

        # TPOT: time per output token (excluding first token)
        if len(tics) > 2:
            for i in range(1, len(tics) - 1):
                tpot_values.append(tics[i + 1] - tics[i])

        # E2E: end-to-end latency
        e2e = tics[-1] - tics[0]
        e2e_values.append(e2e)

        total_tokens += len(tics) - 1  # output tokens

    # Calculate statistics
    def percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)

    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "p99": 0}
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
        }

    # Calculate duration and throughput
    min_start = min(r.tics[0] for r in results if r.tics)
    max_end = max(r.tics[-1] for r in results if r.tics)
    duration = max_end - min_start

    return {
        "num_requests": len(results),
        "total_tokens": total_tokens,
        "duration_seconds": duration,
        "request_throughput": len(results) / duration if duration > 0 else 0,
        "token_throughput": total_tokens / duration if duration > 0 else 0,
        "ttft_seconds": stats(ttft_values),
        "tpot_seconds": stats(tpot_values),
        "e2e_latency_seconds": stats(e2e_values),
    }


if __name__ == "__main__":
    asyncio.run(main())
