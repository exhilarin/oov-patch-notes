# Full pipeline:  make all      (data -> splits -> all models -> benchmark)
PY ?= python
SRC := cd src &&

.PHONY: all data splits classical bilstm transformers zeroshot benchmark clean

all: data splits classical bilstm transformers zeroshot benchmark

data:            ## 1. collect raw announcements from Steam (needs internet, ~1 h) and clean them
	$(PY) src/collect.py --pages 5
	$(PY) src/clean.py

splits:          ## 2. random 80/10/10 split + held-out unseen-games test split
	$(SRC) $(PY) split.py

classical:       ## 3a. keyword rules and CRF (CPU)
	$(SRC) $(PY) train_classical.py --train-size 8000

bilstm:          ## 3b. BiLSTM-CRF from scratch, 3 seeds (GPU recommended)
	$(SRC) $(PY) train_bilstm_crf.py --train-size 15000 --epochs 6

transformers:    ## 3c. BERT (baseline), DeBERTa-v3 and ModernBERT (new) fine-tuning (GPU)
	$(SRC) $(PY) train_transformer.py --model bert --train-size 20000
	$(SRC) $(PY) train_transformer.py --model deberta --train-size 20000 --batch-size 2 --accum 4 --grad-ckpt --freeze-embeddings
	$(SRC) $(PY) train_transformer.py --model modernbert --train-size 20000 --batch-size 2 --accum 4 --grad-ckpt

zeroshot:        ## 3d. CRF spans / gold spans + Laya zero-shot typing (needs models/crf.pkl)
	$(SRC) $(PY) zero_shot_laya.py

benchmark:       ## 4. results/benchmark.md + figures, results/error_analysis.md
	$(SRC) $(PY) benchmark.py
	$(SRC) $(PY) error_analysis.py

clean:
	rm -rf .cache models
