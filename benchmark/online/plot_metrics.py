"""
Generate time-series plots from saved metrics JSON data.

Usage:
    python plot_metrics.py <path_to_metrics_samples.json> [--output-dir <dir>]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Color palette for different metric categories
COLORS = {
    "request_counts": ["#2E86AB", "#E94F37", "#44AF69"],
    "token_usage": ["#F6BD60"],
    "kv_cache_tokens": ["#845EC2", "#D65DB1"],
    "kv_cache_ratio": ["#FF9671"],
    "throughput": ["#00C9A7"],
    "cache_hit_rate": ["#9B5DE5"],
}

# Metric groupings for combined plots
# Note: Metrics in the same group should have similar scales for meaningful visualization
METRIC_GROUPS = {
    "request_counts": [
        ("minisgl_running_requests", "Running", "gauge"),
        ("minisgl_queued_requests", "Queued", "gauge"),
        ("minisgl_completed_requests", "Completed", "counter"),
    ],
    "token_usage": [
        ("token_usage_raw", "Total Tokens", "counter"),
    ],
    "kv_cache_tokens": [
        ("minisgl_num_used_tokens", "Used Tokens", "gauge"),
        ("minisgl_max_total_num_tokens", "Max Capacity", "gauge"),
    ],
    "kv_cache_ratio": [
        ("minisgl_token_usage", "Usage Ratio", "gauge"),
    ],
    "throughput": [
        ("minisgl_output_throughput", "Throughput (tok/s)", "gauge"),
    ],
    "cache_hit_rate": [
        ("minisgl_cache_hit_rate", "Hit Rate", "gauge"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from JSON data")
    parser.add_argument("input_file", type=str, help="Path to metrics samples JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: same as input file)")
    parser.add_argument("--title", type=str, default="Metrics Time Series", help="Plot title")
    parser.add_argument("--style", type=str, default="seaborn-v0_8", help="Matplotlib style (default: seaborn-v0_8)")
    return parser.parse_args()


def plot_metrics(input_file: str, output_dir: Path, title: str, style: str):
    """Generate time-series plots from metrics samples."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("Error: matplotlib is not installed. Install it with: pip install matplotlib")
        return

    # Set style
    plt.style.use(style)

    # Load data
    with open(input_file, "r") as f:
        samples = json.load(f)

    if not samples:
        print("No data to plot")
        return

    # Extract timestamps
    timestamps = [s.get("timestamp", i) for i, s in enumerate(samples)]

    # Pre-calculate token_usage_raw if not present
    for s in samples:
        if "token_usage_raw" not in s:
            s["token_usage_raw"] = s.get("minisgl_total_input_tokens", 0) + s.get("minisgl_total_output_tokens", 0)

    # Filter to groups that have data
    available_groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for group_name, metrics in METRIC_GROUPS.items():
        available_metrics = []
        for metric, display_name, metric_type in metrics:
            if metric in samples[0] or metric == "token_usage_raw":
                available_metrics.append((metric, display_name, metric_type))
        if available_metrics:
            available_groups[group_name] = available_metrics

    if not available_groups:
        print("No recognized metrics found in data")
        print("Available keys:", list(samples[0].keys()))
        return

    # Calculate number of subplots (one per group)
    num_groups = len(available_groups)

    # Create figure with appropriate size
    fig, axes = plt.subplots(num_groups, 1, figsize=(14, 5 * num_groups))
    if num_groups == 1:
        axes = [axes]

    # Plot each group
    for idx, (group_name, metrics) in enumerate(available_groups.items()):
        ax = axes[idx]
        group_colors = COLORS.get(group_name, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

        for metric_idx, (metric, display_name, metric_type) in enumerate(metrics):
            values = [s.get(metric, 0) for s in samples]
            color = group_colors[metric_idx % len(group_colors)]

            ax.plot(
                timestamps, values,
                linewidth=2,
                color=color,
                label=display_name,
                marker='o' if len(samples) <= 20 else '',
                markersize=4,
                alpha=0.9
            )

        # Set labels and title
        group_titles = {
            "request_counts": "Request Counts",
            "token_usage": "Token Usage",
            "kv_cache_tokens": "KV Cache Token Counts",
            "kv_cache_ratio": "KV Cache Usage Ratio",
            "throughput": "Throughput",
            "cache_hit_rate": "Cache Hit Rate",
        }
        ax.set_title(group_titles.get(group_name, group_name), fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        # Add light background for chart area
        ax.set_facecolor('#FAFAFA')

    # Set main title
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save combined plot
    input_path = Path(input_file)
    output_file = output_dir / f"{input_path.stem}_plot.png"
    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    print(f"Saved combined plot to {output_file}")

    # Save individual metric plots (optional - can be disabled)
    save_individual = False  # Set to True to also save individual plots
    if save_individual:
        for metric, display_name, metric_type in METRIC_GROUPS.get("request_counts", []):
            values = [s.get(metric, 0) for s in samples]
            if metric not in samples[0]:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(timestamps, values, linewidth=2, color='#2E86AB', marker='o' if len(samples) <= 20 else '')
            ax.set_title(display_name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))
            ax.set_facecolor('#FAFAFA')

            plt.tight_layout()
            individual_file = output_dir / f"{metric}_plot.png"
            plt.savefig(individual_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved {metric} plot to {individual_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for group_name, metrics in available_groups.items():
        print(f"\n[{group_titles.get(group_name, group_name)}]")
        for metric, display_name, metric_type in metrics:
            values = [s.get(metric, 0) for s in samples]
            if values:
                sorted_vals = sorted(values)
                print(f"\n  {display_name}:")
                print(f"    Min:    {min(values):.6f}")
                print(f"    Max:    {max(values):.6f}")
                print(f"    Mean:   {sum(values) / len(values):.6f}")
                if len(sorted_vals) > 5:
                    median_idx = len(sorted_vals) // 2
                    p90_idx = int(len(sorted_vals) * 0.9)
                    p99_idx = min(int(len(sorted_vals) * 0.99), len(sorted_vals) - 1)
                    print(f"    Median: {sorted_vals[median_idx]:.6f}")
                    print(f"    P90:    {sorted_vals[p90_idx]:.6f}")
                    print(f"    P99:    {sorted_vals[p99_idx]:.6f}")


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(args.input_file, output_dir, args.title, args.style)


if __name__ == "__main__":
    main()
