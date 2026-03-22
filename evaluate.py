"""
全新的evaluate，支持模型的acc评估

python evaluate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
from pathlib import Path
from python.minisgl.core import SamplingParams
from python.minisgl.llm import LLM

DEFAULT_EVAL_FILE = Path(__file__).parent / "ceval_subset.jsonl"
ACCURACY_DROP_LIMIT = 0.05

def load_eval_data(eval_file: str) -> list:
    """
    从数据集文件加载数据，返回 list data
    """
    data = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"已加载 {len(data)} 道评测题（来自 {eval_file}）")
    return data

def build_prompt(item: dict) -> str:
    """
    构建评测题目的 prompt
    """
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
    """
    从输出内容中提取ABCD的选项
    """
    for ch in output.strip():
        if ch.upper() in ("A", "B", "C", "D"):
            return ch.upper()
    return "X"

def run_accuracy_eval(llm: LLM, eval_data: list, batch_size: int = 16) -> dict:
    """
    逐题推理
    """
    print(f"\n 开始精度评测，共 {len(eval_data)} 道题")
    print("-" * 60)
    
    correct = 0
    wrong_cases = []
    t_start = time.perf_counter()
    sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
    
    # 按 batch 处理数据
    for batch_idx in range(0, len(eval_data), batch_size):
        batch = eval_data[batch_idx : batch_idx + batch_size]
        prompts = [build_prompt(item) for item in batch]
        
        # 推理
        results = llm.generate(prompts, sampling_params)
        
        for item, result in zip(batch, results):
            pred = extract_answer(result["text"])
            gold = item["answer"].upper()
            is_correct = (pred == gold)
            
            if is_correct:
                correct += 1
            else:
                wrong_cases.append({
                    "id": item.get("id", batch_idx),
                    "question": item["question"][:60] + "...",
                    "pred": pred,
                    "gold": gold,
                })
        
        # 这部分需要放在 for 循环内
        current_count = min(batch_idx + batch_size, len(eval_data))
        if (current_count) % 20 == 0 or current_count == len(eval_data):
            acc_so_far = correct / current_count
            print(
                f"[{current_count:4d}/{len(eval_data)}]"
                f"当前准确率： {acc_so_far*100:.1f}%"
                f"正确：{correct}  错误：{current_count - correct}"
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
        "wrong_cases": wrong_cases[:5],
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
    
def parse_args():
    parser = argparse.ArgumentParser(description="精度/准确率评测")
    parser.add_argument(
        "--model_name", type=str, default="/inspire/ssd/project/mianxiangdayuyanmoxing/public/Qwen2.5-14B-Instruct",
        help="模型名称（HuggingFace 或本地路径），例如 Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--eval_file", type=str, default=str(DEFAULT_EVAL_FILE),
        help="评测数据集路径（默认：ceval_subset.jsonl）"
    )
    parser.add_argument(
        "--baseline_acc", type=float, default=None,
        help="基线准确率（0~1），用于判断精度是否达标，例如 --baseline_acc 0.72"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="批处理大小（默认：16，越大越快）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON），例如 sglang_result.json"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    llm = LLM(args.model_name,gpu_id=1)
    eval_data = load_eval_data(args.eval_file)
    print("参数、模型、评测数据加载成功")
    
    #评测
    result = run_accuracy_eval(llm, eval_data, batch_size=args.batch_size)
    print_accuracy_result(result, args.baseline_acc)
    
    #保存
    if args.output:
        # wrong_cases 含 list，序列化前过滤
        out = {k: v for k, v in result.items() if k != "wrong_cases"}
        out["wrong_cases_count"] = result["wrong"]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] 结果已保存至: {args.output}")