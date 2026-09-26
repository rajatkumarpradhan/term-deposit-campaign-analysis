# Pre-call term deposit targeting

> **Historical offline ML case study** · Public UCI data · No live customers, outreach system or financial eligibility decision.

**Start here:** [Run the analysis](#run-it) · [Method and leakage boundary](#business-framing-and-method) · [Observed results](#observed-results) · [Interview discussion](#interview-discussion)


An end-to-end, interview-ready machine-learning case study using the public UCI Bank Marketing dataset (Portuguese bank, May 2008–November 2010). The question: **with capacity to call only 20% of customers, can we rank likely subscribers using only information plausibly known before a new call?**

**Portfolio owner:** Rajat Kumar. This is a historical ML demonstration, not an outreach or banking eligibility system.

## Run it

Python 3.10+ recommended (the declared dependency ranges need a compatible Python environment). The source data is bundled so the run needs no account, API key or network connection.

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python analysis.py
jupyter notebook project.ipynb             # optional guided walkthrough
```

`analysis.py` overwrites reproducible results in `outputs/`: metrics, holdout scores, five PNG charts, a validation-only feature-importance table, and a saved model. Run from any working directory. No customer identifiers are included in the UCI CSV. `source_row` is only the original zero-based row index. On first run allow about a minute for installation/training. Tested locally with Python 3.10, pandas 1.5.3, scikit-learn 1.6.1.

### Batch ranking demo

After running `analysis.py`, score a compatible CSV containing the 10 pre-call fields with:

```bash
python predict.py --input 9-bank-additional-full.csv --output outputs/demo_queue.csv --capacity 0.2
```

The demo ranks rows within this batch and flags the highest-scoring 20% for **review only**. Scores are not calibrated probabilities, and the CSV should not be used to contact people. The saved `model.joblib` is generated locally by `analysis.py`; load only models you generated or otherwise trust (joblib uses pickle).

## Business framing and method

- **Outcome:** `y=yes`, term-deposit subscription. **Action:** prioritize calls. This is a ranking experiment, not a demonstration of increased revenue or campaign causality.
- **Data:** 41,188 rows, 20 input variables and one outcome; supplied file `bank-additional-full.csv` is ordered by date. The schema and provenance are described by [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306). Citation: Moro, S., Rita, P. & Cortez, P. (2014), *Bank Marketing*, UCI Machine Learning Repository. The underlying paper is *A data-driven approach to predict the success of bank telemarketing*.
- **Audit and cleaning:** assert schema/row count/outcome values. Original CSV has no pandas nulls, but literal `unknown` is a category (not zero missing data), and `pdays=-1` means no prior contact; convert that sentinel to missing and add an imputation indicator within the training-only pipeline. There are 12 fully duplicated rows; keep them because no reliable customer identity is available to decide whether they are repeat contacts. Numeric median imputation, standardization and categorical one-hot encoding all fit on the training partition only.
- **Leakage boundary:** exclude `duration`, explicitly flagged by UCI as only known after the call; also exclude contact, month, day, current-campaign contact count and all contemporaneous socioeconomic columns to keep the pre-call claim conservative. Included fields are age, job, marital, education, default, housing, loan and prior-campaign history (`pdays`, `previous`, `poutcome`). Some of these may still need operational availability verification at deployment.
- **Explainability:** grouped permutation importance shuffles each original feature on a fixed 1,200-row validation sample (three repeats) and measures the change in average precision. The chart shows model reliance, not causal effects; correlations can hide importance. The test set is not used for interpretation or selection.
- **Validation:** positional chronological split of the UCI file: first 70% train, next 15% validation, last 15% untouched test. Compare class-weighted logistic regression and histogram gradient boosting. Pick once by validation average precision (PR-AUC), then assess test. The 20% capacity is declared up front. A validation-derived score threshold is also applied to the test to show that a fixed threshold does not preserve capacity under drift. No tuning on test.

## Observed results

| Measure | Validation | Latest holdout |
| --- | ---: | ---: |
| Subscription prevalence | 10.65% | 38.45% |
| Chosen model | Histogram gradient boosting | Same model, trained on initial 70% |
| PR-AUC | 0.1899 | 0.4547 |
| ROC-AUC | 0.6198 | 0.5843 |
| Precision in highest-scoring 20% | 19.58% | 46.76% |
| Lift versus that partition's overall rate | 1.84x | 1.22x |
| Subscribers captured in highest-scoring 20% | 36.78% | 24.33% |

At a fixed 20% call capacity on the test period, the ranker selected 1,236 of 6,179 rows and found 578 of 2,376 subscribers. A validation-selected score cutoff instead selected **3,131** holdout rows (50.7%, not 20%): temporal distribution shifts make a fixed score cutoff unreliable. Do not compare PR-AUC across partitions without noting the much higher test prevalence. The holdout ranking lift is modest, and model calibration is weak (test Brier score 0.309); raw probabilities should not be used as expected revenue.

## Interview discussion

1. Start with the decision and constraint: a top-20% calling queue, not a generic accuracy contest.
2. Show why duration is leakage and why a shuffled split would flatter deployment performance on time-ordered campaigns.
3. Walk through the EDA, the fit-on-train preprocessing, both baselines, validation selection and the untouched holdout. The ROC curve alone hides low-base-rate decision costs; include PR-AUC, lift and captured positives.
4. Point out the hard result, not a polished fiction: outcome rate jumps from 5.57% in training to 38.45% on holdout. A constant probability threshold triples the intended contact volume. Use per-batch ranking rather than uncalibrated cutoffs, monitor prevalence/calibration and retrain only after a new forward validation.
5. Ask about contact cost, deposit margin, frequency limits, consent rules, and segmentation before claiming ROI. A prospective holdout/A-B test is needed to estimate *incremental* benefit. The dataset has no customer IDs for identity-level deduplication, grouping or customer-level fairness audit.

## Files

- `project.ipynb` - executable guided notebook, including the generated charts and actual test metrics.
- `analysis.py` - complete deterministic pipeline and chart generator.
- `9-bank-additional-full.csv` - source data unchanged from UCI archive.
- `outputs/metrics.json` - unrounded metrics and split details.
- `outputs/test_scores.csv` - all 6,179 holdout scores, actual outcomes and top-20% flag.
- `outputs/*.png` - EDA, precision-recall/calibration, confusion matrix, decile analysis, and validation-only feature importance.
- `outputs/permutation_importance.json` - feature reliance estimates from validation data.
- `outputs/model.joblib` and `predict.py` - local model artifact and batch-ranking demo; no API key required.
- `REPORT.md` - story, diagnostics, limitations and follow-up plan.

This is an analysis of a historical Portuguese campaign, not a production recommendation for a particular bank or any individual's credit/financial eligibility.
