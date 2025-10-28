# Makefile — One command to run all evals
#路径
#Rundong Guo
# BASE_PATH = /Users/rundongguo/Desktop/cogs\ 108/Untitled/Small-Data

#iris
BASE_PATH = /Users/iriswu/Desktop/3001\ Small\ Data/Small-Data


OUTDIR    = $(BASE_PATH)/reports/baselines
EXPER_DIR = experiments

SIZES = 50 200

# eval:
# 	@for size in $(SIZES); do \
# 		echo ">>> Evaluating train_$$size.csv..."; \
# 		python -m evaluation.eval_calls \
# 			--gold $(BASE_PATH)/data/splits/train_$$size.csv \
# 			--pred $(EXPER_DIR)/pred_train_$$size.csv \
# 			--run_name train_$$size \
# 			--out_dir $(OUTDIR); \
# 		echo ""; \
# 	done

eval:
	@for size in $(SIZES); do \
		echo ">>> Evaluating train_$$size.csv..."; \
		python -m evaluation.eval_calls \
			--gold $(BASE_PATH)/splits/train_$$size.csv \
			--pred $(EXPER_DIR)/pred_train_$$size.csv \
			--run_name train_$$size \
			--out_dir $(OUTDIR); \
		echo ""; \
	done


# ==== Zero-shot Baseline (Task 5) ============================================

# 覆盖用：make zero_shot_all MODEL=openai:gpt-4o-mini SPLIT=splits/test.csv
MODEL ?= openai:gpt-4o-mini
SPLIT ?= splits/test.csv

ZOUT   := experiments/zero_shot
PRED   := $(ZOUT)/$(subst :,_,$(MODEL))/pred_test.csv
RUN    := zero_shot_$(subst :,_,$(MODEL))
OUTDIR_Z := reports/baselines

.PHONY: zero_shot zero_shot_fixcols zero_shot_eval zero_shot_all zero_shot_clean

# 1) 生成预测 CSV：experiments/zero_shot/<model>/pred_test.csv
zero_shot:
	@mkdir -p "$(ZOUT)"
	@export PYTHONPATH="$(PWD):$${PYTHONPATH:-}"; \
	python -m baselines.zero_shot \
	  --split "$(SPLIT)" \
	  --model "$(MODEL)" \
	  --out_dir "$(ZOUT)" \
	  --max_new_tokens 256 \
	  --temperature 0

# 2) 兜底：如缺评测必需列则用 gold 自动补齐
zero_shot_fixcols:
	@python scripts/fix_pred_cols.py --pred "$(PRED)" --gold "$(SPLIT)"

# 3) 评测：输出到 reports/baselines/，并汇总到 summary.md
zero_shot_eval:
	@export PYTHONPATH="$(PWD):$${PYTHONPATH:-}"; \
	python -m evaluation.eval_calls \
	  --gold "$(SPLIT)" \
	  --pred "$(PRED)" \
	  --run_name "$(RUN)" \
	  --out_dir "$(OUTDIR_Z)"

# 一键：预测 -> 补列 -> 评测
zero_shot_all: zero_shot zero_shot_fixcols zero_shot_eval

# 可选：清理该模型产物
zero_shot_clean:
	@rm -rf "$(ZOUT)/$(subst :,_,$(MODEL))"
	@rm -f  "$(OUTDIR_Z)/eval_$(RUN).json"
