# Makefile — One command to run all evals
#路径
#Rundong Guo
BASE_PATH = /Users/rundongguo/Desktop/cogs\ 108/Untitled/Small-Data


OUTDIR    = $(BASE_PATH)/reports/baselines
EXPER_DIR = experiments

SIZES = 50 200 1000 5000

eval:
	@for size in $(SIZES); do \
		echo ">>> Evaluating train_$$size.csv..."; \
		python -m evaluation.eval_calls \
			--gold $(BASE_PATH)/data/splits/train_$$size.csv \
			--pred $(EXPER_DIR)/pred_train_$$size.csv \
			--run_name train_$$size \
			--out_dir $(OUTDIR); \
		echo ""; \
	done

