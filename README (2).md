# Tiny LoRA Fine-Tuning for Code-Mixed (Spanglish) Sentiment Analysis

**Project 3 of 3 — Deep Learning Mini Projects**
Author: Reyen Lourdes · Course / roll no.: `[fill in]` · Submission: `[date]`

How little data and how few trainable parameters does LoRA need before it beats cheap baselines on Spanish-English (Spanglish) tweet sentiment? This project fine-tunes `xlm-roberta-base` with LoRA (Hu et al., 2021) on 500 and 2,000 labelled tweets at ranks 8 and 16, and compares it with a TF-IDF + logistic regression model and with zero-/few-shot prompting of a small LLM, all on one fixed test set.

**Headline result.** With 2,000 training tweets, LoRA (0.9-1.2M trainable parameters, 0.3-0.4% of the model) beats TF-IDF trained on the same 2,000 tweets by about 3-4 macro-F1 points, and roughly matches TF-IDF trained on 12,344 tweets. With 500 tweets, LoRA and TF-IDF are statistically indistinguishable. Training only the classification head on a frozen encoder did not learn at all under the same settings. All results are single-run; see [Limitations](#7-limitations).

---

## 1. Research question

> How few training examples and how few trainable parameters are needed before LoRA beats (a) TF-IDF + logistic regression and (b) zero-/few-shot prompting on code-mixed sentiment?

Scope was deliberately trimmed to fit a free Colab T4 and about one day of work: two ranks × two training-set sizes, one run each, plus a head-only control.

## 2. Data

**Source:** SemEval-2020 Task 9 (SentiMix), Spanglish track (Patwa et al., 2020), the organisers' release on Zenodo (record 3974927). Each tweet has a sentiment label (positive / neutral / negative) and word-level language tags.

| Step | Count |
|---|---|
| Parsed from released `Spanglish_train.conll` / `Spanglish_dev.conll` | 11,623 + 2,905 |
| After removing empty and duplicate texts | 14,344 |
| Test set (fixed, stratified) | 1,500 |
| Validation set (fixed, stratified; epoch selection and TF-IDF `C`) | 500 |
| Remaining pool | 12,344 |
| Training subsets (stratified; the 500 set is nested inside the 2,000 set) | 2,000 / 500 |

Class balance is the same in every split (positive ≈ 50%, neutral ≈ 33%, negative ≈ 17%). The official test labels were not used; the test set is carved out of train + dev with a fixed seed (42), and duplicates were removed first to avoid leakage.

**Reference points.** Always predicting "positive" gives accuracy ≈ 0.50 and macro-F1 ≈ 0.22.

## 3. Methods

**Baselines**
- **TF-IDF + logistic regression:** character n-grams (2-5, word-boundary) plus word n-grams (1-2); `class_weight="balanced"`; `C` ∈ {1, 5, 20} chosen on the validation set separately for each training size. Also trained on the full 12,344-tweet pool as a "more data" reference.
- **Zero-shot and 6-shot `Qwen2.5-0.5B-Instruct`:** the prompt asks for one word (positive / negative / neutral) and the prediction is the highest-scoring of the three label words at the next token, which avoids output-parsing failures. The 6 shots (2 per class) come from the training pool, never the test set.

**LoRA**
- Base model `xlm-roberta-base` (278.9M parameters) with a sequence-classification head.
- LoRA on the attention `query` and `value` projections; `alpha = 2r`, dropout 0.1, ranks r ∈ {8, 16}.
- AdamW, lr 3e-4, weight decay 0.01, batch size 16, max length 96 tokens, 10% warm-up then linear decay, fp16 autocast, gradient clipping 1.0.
- 10 epochs for n = 500 and 6 epochs for n = 2,000. The best epoch is chosen by **validation** macro-F1, and the test set is scored once.
- **Head-only control (r = 0):** encoder frozen, only the classification head trains, same hyper-parameters.
- Trainable parameters are reported as adapters only (`lora_params`) and adapters + head (`trainable_params`), since the head alone is ≈ 0.59M parameters.

**Evaluation:** macro-F1 (primary, because classes are imbalanced), accuracy, and per-class F1 on the fixed 1,500-tweet test set; a paired bootstrap (1,000 resamples of the test set) for LoRA-minus-TF-IDF differences; error analysis by confusion matrix and by degree of code-mixing.

## 4. Results

### 4.1 Main table (test set, n = 1,500)

| Method | Train size | Trainable params | Macro-F1 | Acc. | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|---|
| Zero-shot Qwen2.5-0.5B | 0 | 0 | 0.355 | 0.411 | 0.254 | 0.504 | 0.307 |
| 6-shot Qwen2.5-0.5B | 6 | 0 | 0.386 | 0.385 | 0.421 | 0.381 | 0.358 |
| TF-IDF + LR | 500 | 51,939 ¹ | 0.382 | 0.451 | 0.238 | 0.330 | 0.578 |
| TF-IDF + LR | 2,000 | 158,574 ¹ | 0.424 | 0.475 | 0.321 | 0.354 | 0.598 |
| TF-IDF + LR | 12,344 | 575,565 ¹ | 0.465 | 0.497 | 0.395 | 0.396 | 0.602 |
| XLM-R head-only (r = 0) | 500 | 592,899 | 0.225 | 0.501 | 0.000 | 0.008 | 0.667 |
| XLM-R head-only (r = 0) | 2,000 | 592,899 | 0.225 | 0.501 | 0.000 | 0.008 | 0.668 |
| XLM-R + LoRA r = 8 | 500 | 887,811 | 0.400 | 0.516 | 0.138 | 0.402 | 0.659 |
| XLM-R + LoRA r = 16 | 500 | 1,182,723 | 0.356 | 0.514 | 0.132 | 0.271 | 0.667 |
| XLM-R + LoRA r = 8 | 2,000 | 887,811 | 0.456 | 0.529 | 0.349 | 0.360 | 0.660 |
| XLM-R + LoRA r = 16 | 2,000 | 1,182,723 | **0.463** | **0.536** | 0.423 | 0.302 | 0.662 |

¹ Number of logistic-regression weights (features × 3 + 3); not comparable to neural-network parameters.
Adapter-only parameters: r = 8 → 294,912; r = 16 → 589,824 (plus the 592,899-parameter head).

![Trainable parameters vs macro-F1](results/params_vs_f1.png)

*Figure 1: macro-F1 against trainable parameters (log scale). Dashed lines: TF-IDF at the same training size; dotted grey lines: zero-/few-shot LLM.*

### 4.2 Does LoRA beat TF-IDF at the same training size? (paired bootstrap)

Difference in test macro-F1 (LoRA minus TF-IDF), median and 95% interval over test-set resamples.

| Train size | Model | Difference | 95% interval | Verdict |
|---|---|---|---|---|
| 500 | LoRA r = 8 | +0.018 | [-0.018, +0.051] | inconclusive |
| 500 | LoRA r = 16 | -0.026 | [-0.060, +0.008] | inconclusive |
| 500 | Head-only | -0.157 | [-0.186, -0.130] | TF-IDF better |
| 2,000 | LoRA r = 8 | +0.032 | [+0.002, +0.063] | LoRA better |
| 2,000 | LoRA r = 16 | +0.039 | [+0.009, +0.070] | LoRA better |
| 2,000 | Head-only | -0.198 | [-0.225, -0.171] | TF-IDF better |

This interval reflects test-set sampling noise only, not variation between training runs.

### 4.3 Findings

1. **Data threshold.** At n = 500 LoRA does not reliably beat TF-IDF (+0.018 and -0.026, both intervals include 0). At n = 2,000 both ranks beat TF-IDF at the same size by 3-4 points, though the intervals only just exclude zero.
2. **LoRA at 2,000 examples ≈ TF-IDF at 12,344.** Macro-F1 0.456 / 0.463 against 0.465, i.e. about six times less labelled data for similar macro-F1.
3. **Rank barely matters.** At n = 2,000, r = 8 and r = 16 differ by 0.007 macro-F1. At n = 500, r = 8 scored higher (0.400 vs 0.356), but the r = 16 validation curve was still rising at epoch 10, so that gap more likely reflects under-training than a real rank effect.
4. **The adapters do the learning.** The head-only control stayed at the majority-class solution (macro-F1 0.225, accuracy 0.501, F1 of exactly 0 for negative) at both sizes. This holds for these hyper-parameters; a better-tuned head on frozen features might do better.
5. **Few-shot prompting with a 0.5B model is weak.** 6-shot reaches 0.386, similar to TF-IDF at n = 500 (0.382) and LoRA r = 8 at n = 500 (0.400). I did not test these differences for significance.
6. **Accuracy flatters LoRA.** LoRA's accuracy (0.52-0.54) is higher than TF-IDF's (0.45-0.50), but much of that comes from predicting the majority "positive" class more often (positive F1 ≈ 0.66 against ≈ 0.58-0.60). TF-IDF used balanced class weights; the LoRA runs did not. Macro-F1 is the fairer comparison.
7. **Absolute scores are low.** The best macro-F1 is 0.463. The best SemEval-2020 Spanglish system reached 80.6 F1 using the full training set and the task's own scoring, so the numbers are not comparable. They show what a tiny training budget buys, not what the task allows.

**Answer to the research question.** In this setup, LoRA first clearly beats the TF-IDF baseline at about 2,000 labelled examples with roughly 0.9-1.2M trainable parameters (0.3-0.4% of XLM-R base). At 500 examples it is on par with TF-IDF and with 6-shot prompting. Under 0.6M trainable parameters spent on the head alone, nothing is learned.

## 5. Error analysis

Best LoRA run by validation macro-F1: **r = 8, n = 2,000** (validation 0.480, test 0.456). 706 of 1,500 test tweets are misclassified.

### 5.1 Confusion matrices (rows = true, columns = predicted)

LoRA r = 8, n = 2,000

| | pred neg | pred neu | pred pos |
|---|---|---|---|
| **true neg** (260) | 75 | 78 | 107 |
| **true neu** (489) | 46 | 155 | 288 |
| **true pos** (751) | 49 | 138 | 564 |

TF-IDF, n = 2,000

| | pred neg | pred neu | pred pos |
|---|---|---|---|
| **true neg** (260) | 86 | 71 | 103 |
| **true neu** (489) | 93 | 161 | 235 |
| **true pos** (751) | 97 | 189 | 465 |

- The largest error cell for both models is **neutral predicted as positive** (LoRA 288 of 489 = 59%; TF-IDF 235 of 489 = 48%).
- Recall by class (LoRA / TF-IDF): negative 0.29 / 0.33, neutral 0.32 / 0.33, positive 0.75 / 0.62. LoRA's gain comes mostly from the positive class.
- The negative class (17% of the data) is hard for every model, including LoRA at n = 500 (F1 ≈ 0.13).

### 5.2 Performance by degree of code-mixing

Tweets are bucketed by the minority-language share of language-tagged tokens (0 = one language only, 0.5 = even mix).

| Bucket | Tweets | LoRA macro-F1 | TF-IDF macro-F1 |
|---|---|---|---|
| near-monolingual (≤ 0.05) | 219 | 0.459 | 0.408 |
| light mixing (0.05-0.25) | 847 | 0.444 | 0.423 |
| heavy mixing (> 0.25) | 434 | 0.472 | 0.422 |

LoRA is ahead in every bucket, and neither model degrades on heavily mixed tweets. Bucket sizes are small and I did not test these gaps for significance, so I read this as "no sign that mixing hurts", not as a measured effect.

### 5.3 Hand-labelled error types

`[TODO: fill in after labelling the 50-error sample in error_analysis_to_label.csv]`

| Error type | Count (of 50) | Example (add 1-2 tweets) |
|---|---|---|
| Polarity carried by a word in the minority language | | |
| Non-standard spelling / slang / Spanglish coinage | | |
| Negation across languages | | |
| Sarcasm or irony | | |
| Mixed polarity in one tweet | | |
| Neutral vs. polar boundary | | |
| Mentions / hashtags / URLs carry the content | | |
| Label looks wrong or ambiguous | | |

`[TODO: 3-4 sentences on what the counts show, e.g. whether most errors sit at the neutral boundary, and whether code-switching at the sentiment word matters.]`

## 6. Training behaviour

- Several runs sit at validation macro-F1 ≈ 0.22 (the majority-class solution) for the first few epochs before learning starts. Examples: r = 8 at n = 500 first moves at epoch 5, and r = 16 at n = 500 reaches only 0.344 by epoch 10.
- Validation macro-F1 fluctuates between epochs, partly because the validation set has only 500 tweets. Picking the best epoch on it is therefore noisy.
- Together these suggest the n = 500 runs are close to under-trained, so the n = 500 conclusions are the weakest.

## 7. Limitations

- **Single run per configuration.** No seed variance is measured. The bootstrap covers test-set noise only, and differences below about 2 points should be treated as inconclusive.
- **Possible under-training at n = 500** (see section 6); more epochs or a higher learning rate were not explored, and hyper-parameters were not tuned per configuration.
- **Head-only control is not a tuned baseline.** It used the LoRA hyper-parameters, so "frozen features don't work" is not established beyond this setting.
- **Class imbalance handled unevenly:** TF-IDF used balanced class weights; the neural runs did not.
- **Weak LLM baseline:** only a 0.5B model with one fixed prompt and one set of 6 examples.
- **Data size discrepancy:** the released files parsed to 11,623 train and 2,905 dev tweets, while the overview papers report 12,002 and 2,998. I did not investigate why (parser handling or the released files), so absolute dataset sizes may differ slightly from the official ones.
- **One data source and period** (tweets, the CALCS-derived Spanglish corpus); results may not transfer to other genres or language pairs.
- **My own test split**, not the official hidden test set, so scores are not directly comparable with the shared-task leaderboard.

## 8. Reproducing

1. Open `project3_lora_sentiment.ipynb` in Google Colab, set Runtime → Change runtime type → **T4 GPU**, and run all cells top to bottom (≈ 1-2 hours in total, mostly the LoRA runs).
2. The notebook downloads the dataset from Zenodo and verifies its checksum. It also uninstalls the preinstalled `torchao`, which conflicts with the newest `peft`.
3. Everything is seeded (`SEED = 42`). Outputs: `results.csv`, `results_table.csv`, `params_vs_f1.png`, `preds.pkl`, `error_analysis_to_label.csv`.

Software: Python 3, PyTorch, Hugging Face `transformers`, `peft`, `scikit-learn`, `pandas`, `matplotlib`.

## 9. References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Patwa, P., Aguilar, G., Kar, S., Pandey, S., PYKL, S., Gambäck, B., Chakraborty, T., Solorio, T., Das, A. (2020). *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.* SemEval-2020. arXiv:2008.04277.
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020 (XLM-R). arXiv:1911.02116.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
- Mangrulkar, S. et al. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.* https://github.com/huggingface/peft
