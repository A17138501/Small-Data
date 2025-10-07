# Small-Data

## 📘 项目简介
本项目属于 **Learning from Small Data (Small-Data)** 课程项目，旨在研究 **LLM 在小样本函数调用（Function Calling）任务中的表现**。  
团队统一使用 Hugging Face 数据集 `xlam-function-calling-60k`（由 APIGen 预处理）进行实验，探索 Prompting、Fine-tuning 和 Heuristic 方法在低样本场景下的性能差异。

---

## 🧩 任务 0：数据下载脚本（Run）

### 📌 为什么需要
我们不将大文件（如原始数据集）上传到 GitHub，而是通过 **Hugging Face API** 动态拉取，保证：
- 仓库轻量；
- 数据源一致；
- 各组员可一键复现。

---

### ⚙️ 工作内容
通过 `load_dataset()` 下载数据并保存为 CSV 文件：

```python
from huggingface_hub import login
from datasets import load_dataset
import pandas as pd, os

# 登录 Hugging Face（只需一次，不保存 Token）
login()

# 加载数据集（当前使用验证可用的数据源）
dataset = load_dataset("minpeter/xlam-function-calling-60k-parsed")

# 保存至本地 data/raw/
os.makedirs("data/raw", exist_ok=True)
pd.DataFrame(dataset["train"]).to_csv("data/raw/apigen.csv", index=False)
print("✅ Saved dataset to data/raw/apigen.csv")
