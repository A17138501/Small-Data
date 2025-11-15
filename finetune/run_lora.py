#!/usr/bin/env python
"""
One-shot LoRA pipeline for Small-Data project (HanDing).

功能：
  1. 用 MLX 在本地跑 LoRA 微调（n_samples = 50 或 200）
  2. 用 finetune.predict_lora 生成预测 CSV
  3. 调用 make eval 生成评测结果
  4. 简单记录训练日志到 reports/finetune/trainlog.md

用法（在仓库根目录 Small-Data 下）：
  conda activate mlx-lora
  cd ~/Desktop/Small-Data
  python run_lora.py --n_samples 50
  或
  python run_lora.py --n_samples 200
"""

import argparse
import os
import subprocess
from pathlib import Path
from datetime import datetime


def run(cmd, env=None):
    """打印并执行命令。"""
    print("\n>>> Running command:")
    print("   ", " ".join(str(c) for c in cmd))
    print()
    subprocess.run(cmd, check=True, env=env)


def prepare_mlx_data(repo_root: Path, n: int) -> Path:
    """
    把 data/sft/train_{n}.jsonl 复制成 MLX 需要的目录结构：
      data/sft/mlx_{n}/train.jsonl
      data/sft/mlx_{n}/valid.jsonl  （这里简单用同一份做 val）
    """
    sft_root = repo_root / "data" / "sft"
    src = sft_root / f"train_{n}.jsonl"
    if not src.exists():
        raise SystemExit(f"[ERROR] 找不到训练数据文件: {src}")

    mlx_dir = sft_root / f"mlx_{n}"
    mlx_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.jsonl", "valid.jsonl"):
        dst = mlx_dir / name
        with open(src, "r", encoding="utf-8") as fin, open(
            dst, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                fout.write(line)

    print(f"[INFO] 已准备 MLX 数据目录: {mlx_dir}")
    print(f"       train.jsonl <- {src}")
    print(f"       valid.jsonl <- {src}")
    return mlx_dir


def append_train_log(repo_root: Path, n: int, text: str) -> None:
    """把关键信息追加写入训练日志。"""
    log_dir = repo_root / "reports" / "finetune"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "trainlog.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {ts} — LoRA n_samples={n}\n")
        f.write(text.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        choices=[50, 200],
        required=True,
        help="训练集大小，只能是 50 或 200。",
    )
    args = parser.parse_args()
    n = args.n_samples

    # 统一切到仓库根目录
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    # 1) 准备 MLX 数据目录
    mlx_dir = prepare_mlx_data(repo_root, n)

    # LoRA adapter 输出目录
    adapter_dir = (
        repo_root
        / "experiments"
        / "finetune"
        / "llama3.2-1b"
        / "mlx_lora"
        / f"train_{n}"
    )
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # 2) LoRA 训练（已经验证过可以在你机器上跑通的配置）
    train_cmd = [
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--train",
        "--data",
        str(mlx_dir),
        "--adapter-path",
        str(adapter_dir),
        "--batch-size",
        "2",
        "--iters",
        "200",
        "--max-seq-length",
        "1024",
        "--learning-rate",
        "2e-5",
        "--save-every",
        "50",
        "--steps-per-eval",
        "50",
        "--val-batches",
        "20",
        # add these options to improve training
        "--mask-prompt",
        "--optimizer",
        "adamw",
    ]
    # 训练过程的环境变量（目前用默认即可）
    env_train = os.environ.copy()
    # uncomment if you want to re-train
    run(train_cmd, env=env_train)

    append_train_log(
        repo_root,
        n,
        text=f"命令: {' '.join(train_cmd)}\n"
        f"LoRA 适配器目录: {adapter_dir}",
    )

    # 3) 预测
    # split_file = repo_root / "splits" / f"train_{n}.csv"
    split_file = repo_root / "splits" / "test.csv"

    if not split_file.exists():
        raise SystemExit(f"[ERROR] 找不到 split 文件: {split_file}")

    pred_csv = adapter_dir / f"pred_{split_file.stem}_{n}.csv"

    env_pred = os.environ.copy()
    # 确保可以 import finetune.predict_lora
    env_pred["PYTHONPATH"] = str(repo_root)

    predict_cmd = [
        "python",
        "-m",
        "finetune.predict_lora",
        "--mlx_model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--mlx_adapter",
        str(adapter_dir),
        "--split",
        str(split_file),
        "--out_csv",
        str(pred_csv),
    ]
    run(predict_cmd, env=env_pred)

    # 4) 评测
    env_eval = os.environ.copy()
    env_eval["PYTHONPATH"] = str(repo_root)
    env_eval["EVAL_NO_AUTOFILL"] = "1"

    run_name = f"ft_llama3.2_1b_mlxlora_{n}"
    eval_cmd = [
        "make",
        "eval",
        f"RUN={run_name}",
        f"PRED={pred_csv}",
        f"GOLD={split_file}",
    ]
    run(eval_cmd, env=env_eval)

    print("\n[Done]")
    print(f"预测文件: {pred_csv}")
    print(
        "评测结果应在 reports/baselines/ 下，文件名类似："
        f"eval_{run_name}.json / eval_{run_name}.md"
    )
    print("训练日志写入: reports/finetune/trainlog.md")


if __name__ == "__main__":
    main()

#python -m evaluation.eval_calls     
# --gold splits/test.csv     
# --pred experiments/finetune/llama3.2-1b/mlx_lora/train_50/pred_test.csv     
# --run_name ft_llama3.2_1b_mlxlora_50     
# --out_dir reports/

#python -m evaluation.eval_calls     
# --gold splits/test.csv     
# --pred experiments/finetune/llama3.2-1b/mlx_lora/train_200/pred_test.csv     
# --run_name ft_llama3.2_1b_mlxlora_50     
# --out_dir reports/