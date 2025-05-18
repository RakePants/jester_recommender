# Jester Rec
## Metrics
Evaluation metrics used:
- MAE
- Precision@k
- Recall@k
- MAP@k
- NDCG@k
- Diversity@k
- Novelty@k
- Serendipity@k
## Running
1. Set PYTHONPATH to src:
```
export PYTHONPATH="$(pwd)/src"
```
2. Train the model:
```
python scripts/train.py \
  --interactions jester-2m/matrix.xlsx \
  --jokes        jester-2m/jokes.xlsx \
  --output-dir   artifacts/ \
  --test-size    0.2 \
  --seed         42
```
3. Evaluate the model:
```
python scripts/evaluate.py \
  --artifacts-dir artifacts/ \
  --output-dir    artifacts/
```
