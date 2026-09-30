# TRANSFORMER RESULT — PhoBERT Fine-tuning on UIT-VSFC

**Đề tài #2** — Sentiment + Topic classification, Trường Đại học Nha Trang, Củ Chi 2026
**Model:** `vinai/phobert-base-v2` (RoBERTa-base for Vietnamese, ~135M params)
**Device:** CPU-only (torch 2.14.0+cpu, transformers 5.17.0) — full fine-tuning, no frozen layers
**Date:** 2026-09-30

---

## 1. Objective

Train two independent PhoBERT classifiers on the UIT-VSFC dataset:

- **Sentiment** (3 classes): Negative / Neutral / Positive
- **Topic** (4 classes): Lecturer / Training_program / Facility / Others

Both are trained with the **identical, pre-approved configuration**, full fine-tuning of all
encoder layers + classification head. The goal is to compare against the Linear SVM baseline
and quantify the gain — especially on the minority classes (Neutral; Others) where the SVM
struggled in the error analysis.

## 2. Data

Input column: **`text_clean`** (already lowercased/normalized by the preprocessing step).
No new cleaning, no augmentation, no subsampling — full training set used.

| Split | Samples | Sentiment (Neg/Neu/Pos) | Topic (Lect/Train/Facil/Oth) |
|-------|---------|--------------------------|------------------------------|
| Train | 11,424  | —                        | —                            |
| Dev   | 1,583   | —                        | —                            |
| Test  | 3,166   | 1409 / 167 / 1590        | 2290 / 572 / 145 / 159       |

Data integrity: `data/raw/` and `data/processed/` were **not modified** — verified by MD5
checksum (9/9 files unchanged) in the post-training validation.

## 3. Token length analysis → max_length

Token-length stats on `text_clean` (PhoBERT BPE tokenizer), `results/transformer/token_length_stats.json`:

| Split | median | p90 | p95 | p99 | max | >64 tok | >128 tok |
|-------|--------|-----|-----|-----|-----|---------|----------|
| Train | 14     | 29  | 36  | 55  | 163 | 0.52%   | 0.01%    |
| Dev   | 13     | 27  | 35  | 52  | 164 | 0.32%   | 0.06%    |
| Test  | 13     | 29  | 37  | 55  | 102 | 0.69%   | 0.0%     |

**Chosen `max_length = 32`.** Rationale: p90 ≈ 29 → 32 covers ~92% of samples fully while keeping
CPU training tractable. Truncation affects only ~8% of samples and clips their tail (the comment
body), which carries less label signal than the head.

## 4. Model

`AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2")`
→ `RobertaForSequenceClassification`: RoBERTa encoder (12 layers, hidden 768) + new linear
classification head (`classifier.dense` + `classifier.out_proj`, randomly initialized — the
pretrained `lm_head` weights are unused, as expected for a fine-tuning load).

- **All encoder parameters are trainable** (full fine-tuning). No frozen layers.
- num_labels: 3 (sentiment) / 4 (topic); label maps saved to `models/transformer/<task>/label_map.json`.

## 5. Training configuration (identical for both tasks)

| Hyperparameter | Value |
|----------------|-------|
| base model     | vinai/phobert-base-v2 |
| input column   | text_clean |
| max_length     | 32 (padding + truncation) |
| epochs         | 2 |
| batch_size     | 64 |
| optimizer      | AdamW (no_decay on bias + LayerNorm) |
| learning_rate  | 2e-5 |
| weight_decay   | 0.01 |
| warmup_ratio   | 0.1 (linear warmup → linear decay) |
| gradient_clip  | 1.0 |
| seed           | 42 |
| loss           | **weighted CrossEntropyLoss** (class weights below) |
| device         | cpu |

Total optimizer steps: 358 (179 batches/epoch × 2 epochs).

## 6. Class weights (loss re-weighting, NOT resampling)

Inverse-frequency weights from `data/processed/class_weights.json`
(`weight = total / (num_classes × count)`, computed on train only):

| Task | Weights |
|------|---------|
| Sentiment | Negative 0.7153 · **Neutral 8.3144** · Positive 0.6749 |
| Topic | Lecturer 0.3498 · Training_program 1.2976 · **Facility 5.7465** · **Others 5.0819** |

The heavy weights on Neutral / Facility / Others push the model to attend to the rare classes —
this is the mechanism expected to fix the SVM's minority-class collapse seen in error analysis.

## 7. Checkpoint selection & TEST protocol

- **Best checkpoint** chosen by **DEV macro F1** (recomputed each epoch); model + tokenizer +
  label_map saved to `models/transformer/<task>/` only when dev macro F1 improves.
- **TEST set used exactly once** — final evaluation of the selected best checkpoint. No
  test-based model selection, no peeking.

## 8. Training history

### Sentiment

| Epoch | train_loss | dev_acc | dev_macroF1 | dev_wF1 | time |
|-------|-----------|---------|-------------|---------|------|
| 1     | 0.6657    | 0.9318  | 0.8360      | 0.9322  | 4460s (~74m) |
| 2     | 0.3582    | 0.9362  | **0.8456**  | 0.9392  | 3495s (~58m) |

→ Best epoch **2** (dev macro F1 = 0.8456). Train log: `results/transformer/sentiment_train.log`.

### Topic

| Epoch | train_loss | dev_acc | dev_macroF1 | dev_wF1 | time |
|-------|-----------|---------|-------------|---------|------|
| 1     | 0.8751    | 0.8547  | 0.7698      | 0.8644  | 4664s (~78m) |
| 2     | 0.4834    | 0.8579  | **0.7714**  | 0.8672  | 3903s (~65m) |

→ Best epoch **2** (dev macro F1 = 0.7714). Train log: `results/transformer/topic_train.log`.

Both tasks show a healthy loss decrease and a modest dev-F1 gain from epoch 1→2 (no overfitting
spike; the dev curve is still rising, consistent with the approved 2-epoch budget).

## 9. DEV results (best checkpoint)

### Sentiment — dev macro F1 = 0.8456, acc = 0.9362

```
              precision    recall   f1-score   support
    Negative     0.9557    0.9475    0.9516       705
     Neutral     0.5392    0.7534    0.6286        73
    Positive     0.9706    0.9429    0.9565       805
```

### Topic — dev macro F1 = 0.7714, acc = 0.8579

```
                  precision    recall   f1-score   support
        Lecturer     0.9730    0.8775    0.9228      1151
Training_program     0.6780    0.8202    0.7424       267
        Facility     0.8481    0.9571    0.8993        70
          Others     0.4336    0.6526    0.5210        95
```

## 10. TEST results (final, single evaluation)

### Sentiment — **acc = 0.9198, macro F1 = 0.8224, weighted F1 = 0.9226**

```
              precision    recall   f1-score   support
    Negative     0.9345    0.9418    0.9381      1409
     Neutral     0.5185    0.6707    0.5849       167
    Positive     0.9627    0.9264    0.9442      1590
```

### Topic — **acc = 0.8506, macro F1 = 0.7673, weighted F1 = 0.8603**

```
                  precision    recall   f1-score   support
        Lecturer     0.9632    0.8694    0.9139      2290
Training_program     0.6756    0.7902    0.7284       572
        Facility     0.8616    0.9448    0.9013       145
          Others     0.4170    0.7107    0.5256       159
```

## 11. Confusion matrices (TEST, rows = true)

### Sentiment (Neg / Neu / Pos)

```
            Neg   Neu   Pos
Negative   1327    56    26
Neutral      24   112    31
Positive     69    48  1473
```

### Topic (Lecturer / Training_program / Facility / Others)

```
                 Lect  Train  Facil  Others
Lecturer         1991    187      6     106
Training_program    58    452     12      50
Facility             2      4    137       2
Others              16     26      4     113
```

## 12. Per-class comparison vs SVM baseline (TEST)

`results/transformer/model_comparison.csv`:

| Task | Metric | SVM | PhoBERT | Δ |
|------|--------|-----|---------|-----|
| sentiment | accuracy | 0.8920 | **0.9198** | +0.0278 |
| sentiment | macro_f1 | 0.7304 | **0.8224** | **+0.0920** |
| sentiment | weighted_f1 | 0.8872 | **0.9226** | +0.0354 |
| sentiment | f1_Negative | 0.9126 | 0.9381 | +0.0255 |
| sentiment | f1_Neutral | 0.3584 | **0.5849** | **+0.2265** |
| sentiment | f1_Positive | 0.9203 | 0.9442 | +0.0239 |
| topic | accuracy | 0.8588 | 0.8506 | −0.0082 |
| topic | macro_f1 | 0.7531 | **0.7673** | +0.0142 |
| topic | weighted_f1 | 0.8602 | 0.8603 | +0.0001 |
| topic | f1_Lecturer | 0.9204 | 0.9139 | −0.0065 |
| topic | f1_Training_program | 0.7167 | 0.7284 | +0.0117 |
| topic | f1_Facility | 0.9184 | 0.9013 | −0.0171 |
| topic | f1_Others | 0.4570 | **0.5256** | **+0.0686** |

## 13. Headline numbers

| | SVM (baseline) | PhoBERT | Δ |
|---|---|---|---|
| **Sentiment acc / macroF1** | 0.8920 / 0.7304 | **0.9198 / 0.8224** | +2.8 / +9.2 pts |
| **Topic acc / macroF1** | 0.8588 / 0.7531 | **0.8506 / 0.7673** | −0.8 / +1.4 pts |

## 14. Interpretation

**Sentiment is a clear win.** PhoBERT beats the SVM on every metric and every class:
+9.2 macro-F1, and the biggest single gain is on **Neutral** (F1 0.358 → 0.585, +0.23; recall
0.30 → 0.67). This is exactly the failure mode the baseline error analysis flagged — the SVM
was defaulting short/under-specified comments away from Neutral, and contextual embeddings +
the 8.3× loss weight on Neutral correct most of that.

**Topic is a trade-off, not a clean win.** PhoBERT's macro F1 improves (+0.014) because it lifts
the two rare classes — **Others** +0.069 and Training_program +0.012 — but accuracy dips slightly
(−0.008) because it loses a little on the two dominant classes (Lecturer −0.007, Facility −0.017).
In other words the model shifted decision mass toward the minority classes at a small cost on the
majority ones. Since the project weighs macro F1 (balanced performance across classes), this is a
net improvement, but it is not the uniform gain seen in sentiment and should be reported honestly.

## 15. Limitations / notes

- **CPU-only training** — ~75–80 min/epoch at batch 64 / max_length 32; total ~5h for both tasks.
  This forced the 2-epoch budget and max_length=32; a GPU would allow longer sequences and more
  epochs (dev F1 was still rising, so the model is likely slightly under-trained).
- **max_length=32 truncates ~8%** of comments at the tail; a few long reviews lose context.
- **Neutral (sentiment) and Others (topic) remain the weakest classes** despite improvement —
  they are inherently ambiguous (mixed/neutral sentiment; off-topic "other" comments). The residual
  errors here are largely annotation-ambiguity, consistent with the earlier error analysis showing
  >75% of SVM errors were also wrong for a second model.
- **Class weighting is aggressive** (Neutral 8.3×, Facility 5.7×) — it buys minority recall at the
  cost of some majority-class precision (visible in the topic Lecturer/Facility F1 dips).

## 16. Validation

`src/transformer/validate.py` — **ALL CHECKS PASSED**:

- Sample counts: train 11424 / dev 1583 / test 3166 ✓
- Tokenizer + checkpoints loadable (135M params each) ✓
- Predictions: 3166 rows each, valid labels, no NaN ✓
- Metrics: dev + test present, no NaN ✓
- Confusion matrices: 3×3 (sentiment) / 4×4 (topic) ✓
- Training history: 2 epochs each ✓
- **Raw data integrity: 9/9 MD5 unchanged** ✓
- Processed data row counts unchanged ✓

## 17. Artifacts

**Models** — `models/transformer/{sentiment,topic}/`: `model.safetensors`, `config.json`,
tokenizer files (`vocab.txt`, `bpe.codes`, `tokenizer_config.json`, `added_tokens.json`),
`label_map.json`.

**Results** — `results/transformer/`:
- `{task}_metrics.json` — config + dev/test metrics + history
- `{task}_training_history.json` — per-epoch loss/dev metrics
- `{task}_classification_report.txt` — dev + test per-class P/R/F1
- `{task}_confusion_matrix.csv`, `{task}_dev_confusion_matrix.csv`
- `{task}_test_predictions.csv` — 3166 rows (id, text, true, predicted, correct, names)
- `{task}_train.log` — batch-level training progress
- `model_comparison.csv` — SVM vs PhoBERT side-by-side
- `token_length_stats.json` — token-length analysis

**Code** — `src/transformer/`: `config.py`, `dataset.py`, `model.py`, `metrics.py`,
`train_task.py`, `evaluate.py`, `validate.py`, `tokenize_stats.py`.

---

**TRANSFORMER STATUS: READY FOR REVIEW**
