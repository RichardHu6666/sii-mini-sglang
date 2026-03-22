#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_accuracy.py
====================
精度评测脚本

使用方式：
  # 评测在线服务
  python evaluate_accuracy.py --host 127.0.0.1 --port 1919 --eval_file ceval_subset.jsonl

  # 评测本地模型（基线）
  python evaluate_accuracy.py --model_path /path/to/model --eval_file ceval_subset.jsonl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import AsyncOpenAI as OpenAI

DEFAULT_EVAL_FILE   = Path(__file__).parent / "ceval_subset.jsonl"
ACCURACY_DROP_LIMIT = 0.05   # 精度损失上限（绝对值）


def load_eval_data(eval_file: str) -> list:
    """加载 C-Eval 格式数据集"""
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"[INFO] 已加载 {len(data)} 道评测题（来自 {eval_file}）")
    return data


def build_prompt(item: dict) -> str:
    return (
        f"以下是一道单选题，请直接回答选项字母（A/B/C/D），不要有任何解释。\n\n"
        f"题目：{item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n"
        f"答案："
    )


def extract_answer(output: str) -> str:
    for ch in output.strip():
        if ch.upper() in ("A", "B", "C", "D"):
            return ch.upper()
    return "X"   # 未能解析时标记为错误


async def run_one_request(client: OpenAI, prompt: str, model: str, max_tokens: int = 10) -> str:
    """Call the online model service and return the generated output."""
    # 服务端强制返回流式响应，需要手动解析
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"ignore_eos": True, "top_k": 1},
        stream=True,
    )

    content = []
    async for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                content.append(delta.content)

    return "".join(content)


async def run_accuracy_eval_online(client: OpenAI, model: str, eval_data: list) -> dict:
    """
    逐题推理并统计准确率（在线服务版本）。
    """
    print(f"\n[Accuracy] 开始精度评测，共 {len(eval_data)} 道题...")
    print("-" * 60)

    correct = 0
    wrong_cases = []
    t_start = time.perf_counter()

    for i, item in enumerate(eval_data):
        prompt = build_prompt(item)
        output = await run_one_request(client, prompt, model, max_tokens=10)
        pred = extract_answer(output)
        gold = item["answer"].upper()
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
        else:
            wrong_cases.append({
                "id": item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred": pred,
                "gold": gold,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率：{acc_so_far*100:.1f}%  "
                f"正确：{correct}  错误：{i+1-correct}"
            )

    t_end = time.perf_counter()
    accuracy = correct / len(eval_data)

    result = {
        "total": len(eval_data),
        "correct": correct,
        "wrong": len(eval_data) - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases": wrong_cases[:10],
    }
    return result


def run_accuracy_eval_baseline(tokenizer, model, eval_data: list) -> dict:
    """
    逐题推理并统计准确率（本地模型基线版本）。
    """
    from baseline_inference import infer_single

    print(f"\n[Accuracy] 开始精度评测，共 {len(eval_data)} 道题...")
    print("-" * 60)

    correct = 0
    wrong_cases = []
    t_start = time.perf_counter()

    for i, item in enumerate(eval_data):
        prompt = build_prompt(item)
        res = infer_single(tokenizer, model, prompt)
        pred = extract_answer(res["output"])
        gold = item["answer"].upper()
        is_correct = (pred == gold)

        if is_correct:
            correct += 1
        else:
            wrong_cases.append({
                "id": item.get("id", i),
                "question": item["question"][:60] + "...",
                "pred": pred,
                "gold": gold,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(eval_data):
            acc_so_far = correct / (i + 1)
            print(
                f"  [{i+1:4d}/{len(eval_data)}]  "
                f"当前准确率：{acc_so_far*100:.1f}%  "
                f"正确：{correct}  错误：{i+1-correct}"
            )

    t_end = time.perf_counter()
    accuracy = correct / len(eval_data)

    result = {
        "total": len(eval_data),
        "correct": correct,
        "wrong": len(eval_data) - correct,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": round(accuracy * 100, 2),
        "eval_time_sec": round(t_end - t_start, 2),
        "wrong_cases": wrong_cases[:10],
    }
    return result


def print_accuracy_result(result: dict, baseline_acc: float = None):
    """格式化打印精度评测结果"""
    print("\n" + "=" * 60)
    print(" 精度评测结果")
    print("=" * 60)
    print(f"  总题数       : {result['total']}")
    print(f"  答对题数     : {result['correct']}")
    print(f"  答错题数     : {result['wrong']}")
    print(f"  准确率       : {result['accuracy_pct']:.2f}%")
    print(f"  评测耗时     : {result['eval_time_sec']} sec")

    if baseline_acc is not None:
        drop = baseline_acc - result["accuracy"]
        status = "达标" if drop <= ACCURACY_DROP_LIMIT else "超标（扣分）"
        print("-" * 60)
        print(f"  基线准确率   : {baseline_acc*100:.2f}%")
        print(f"  精度下降     : {drop*100:.2f}% （上限 {ACCURACY_DROP_LIMIT*100:.0f}%）")
        print(f"  精度约束状态 : {status}")
        if drop > ACCURACY_DROP_LIMIT:
            print("  [警告] 精度损失超过阈值，「优化效果」评分将扣 50%！")

    print("=" * 60)


async def get_model_name(client: OpenAI) -> str:
    """Get model name from the server."""
    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")


def parse_args():
    parser = argparse.ArgumentParser(description="C-Eval 精度评测")

    # Online service mode
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="在线模型服务主机地址（默认：127.0.0.1）"
    )
    parser.add_argument(
        "--port", type=int, default=1919,
        help="在线模型服务端口（默认：1919）"
    )
    parser.add_argument(
        "--api-key", type=str, default="",
        help="API Key（可选）"
    )

    # Local model mode (baseline)
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="模型本地路径（使用本地模型时指定）"
    )

    # Common options
    parser.add_argument(
        "--eval_file", type=str, default=str(DEFAULT_EVAL_FILE),
        help="评测数据集路径（默认：ceval_subset.jsonl）"
    )
    parser.add_argument(
        "--baseline_acc", type=float, default=None,
        help="基线准确率（0~1），用于判断精度是否达标，例如 --baseline_acc 0.72"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 accuracy_baseline.json"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="模型名称（在线服务模式下可选，自动获取如果未指定）"
    )

    return parser.parse_args()


async def main_async():
    args = parse_args()

    # Determine mode: online service or local model
    use_online = args.host is not None and args.port is not None

    eval_data = load_eval_data(args.eval_file)

    if use_online:
        # Online service mode
        async with OpenAI(
            base_url=f"http://{args.host}:{args.port}/v1",
            api_key=args.api_key or "empty"
        ) as client:
            model = args.model_name or await get_model_name(client)
            print(f"[INFO] 使用在线模型服务：{model} @ http://{args.host}:{args.port}")
            result = await run_accuracy_eval_online(client, model, eval_data)
    else:
        # Local model mode (baseline)
        if args.model_path is None:
            print("[ERROR] 请指定 --model_path 或使用 --host/--port 指定在线服务")
            return
        from baseline_inference import load_model
        tokenizer, model = load_model(args.model_path)
        print(f"[INFO] 使用本地模型：{args.model_path}")
        result = run_accuracy_eval_baseline(tokenizer, model, eval_data)

    print_accuracy_result(result, args.baseline_acc)

    if args.output:
        # wrong_cases 含 list，序列化前过滤
        out = {k: v for k, v in result.items() if k != "wrong_cases"}
        out["wrong_cases_count"] = result["wrong"]
        with open(args.output, "w", encoding="utf-8") as f:
            _json = json
            _json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至：{args.output}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
