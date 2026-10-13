# Alzheimer's Risk Screening

An interactive screening tool built on a global demographic and clinical
dataset. Enter a patient record and four trained classifiers score it live,
with panels for where the referral cut-off belongs, what drives the
prediction, and how much the modeling is actually buying.


## Running it

```bash
pip install -r requirements.txt
python train.py          # about 50 seconds, writes artifacts/
streamlit run app.py
```

`python train.py --search` re-runs the full `GridSearchCV` rather than using
the already-selected parameters. It takes 15 to 40 minutes depending on cores.

## The panels

| Panel | Question |
|---|---|
| Score a patient | What is this person's risk, and do the models agree? |
| Set the cut-off | Where should the screening threshold sit, given what a missed case costs? |
| What drives it | Which features carry the signal? |
| Where it tops out | Is more modelling worth it? |

## Approach

**Encoding.** One-hot with `drop_first` for binary and nominal features,
explicit ordinal encoding for the Low/Medium/High scales, binary encoding for
the 20-country high-cardinality column, standard scaling for the four numeric
features. All of it inside a `ColumnTransformer` so encoders are fit on
training folds only and a single raw record can be scored at inference.

**Models.** Logistic regression, decision tree, random forest and a linear SVM,
each tuned by 5-fold stratified `GridSearchCV` on the training split, evaluated
once on a held-out 20%. The SVM is wrapped in `CalibratedClassifierCV` so all
four expose comparable probabilities.

**Metric.** Accuracy and recall are reported, but the recommended model is
chosen on AUC: how well it ranks risk. In screening, missing a case costs far
more than a false alarm, so the referral cut-off is set separately from the
model, in the threshold panel, rather than left at the default 0.50.

## Results

| Model | Best parameters | Accuracy | Recall | AUC |
|---|---|---|---|---|
| Decision Tree | `gini, max_depth=3, min_samples_split=2` | 0.7297 | 0.7363 | 0.7891 |
| Random Forest | `n_estimators=200, max_depth=10, min_samples_split=5` | 0.7275 | 0.6806 | 0.8026 |
| SVM (LinearSVC) | `C=0.1, class_weight=balanced` | 0.7117 | 0.6008 | 0.7894 |
| Logistic Regression | `C=0.01, l1_ratio=0.0` | 0.7114 | 0.5974 | 0.7894 |

All four cluster near 73% against a 58.7% majority-class baseline. `Age`, the
APOE-ε4 indicator and family history are the strongest predictors by both
mutual information and random forest importance, consistent with the clinical
literature.

## Where the ceiling is

Rather than assume the plateau was the algorithm, it was tested:

| Configuration | Accuracy |
|---|---|
| Always answer No | 58.7% |
| Age alone, one variable | 70.9% |
| Age + APOE-ε4 + family history | 71.2% |
| Gradient boosting, all 24 features | 72.7% |
| Best tuned model | 73.0% |

A single variable lands within 2.1 points of the full pipeline, gradient
boosting does not beat a tuned decision tree, and a learning curve from 4,000
to 40,000 rows moves accuracy by 1.1% while flattening. The limit is the
feature set, not the model, so the useful next step is better features
(biomarkers such as CSF amyloid or MRI volumetrics, or longitudinal cognitive
scores) rather than another algorithm.

## Known limitations

- Diagnosis rate rises in flat steps with age and plateaus after 75, which does
  not match how incidence behaves clinically. The dataset appears at least
  partly synthetic, which caps how far these results generalise.
- No biomarker or imaging features, which is the main reason for the ceiling.
- Country is binary-encoded, so individual bit columns are not interpretable on
  their own and are reported summed.

## Attribution

Group project for Chem 277A, UC Berkeley, with Zander Rothering, Shivani
Tijare, Seungho Yoo and Girish Krishna. The original group notebook is in
`notebook/` (included with permission; source:
github.com/Zander-Rothering/Chem277a_Project_II). Its discussion notes one
limitation: categorical encoders were fit on the full dataset. This app closes
that gap with a single training-only pipeline, and adds threshold analysis
and the ceiling study.

Dataset: Alzheimer's Prediction Dataset (Global), Ankit (2025), Kaggle,
https://doi.org/10.34740/KAGGLE/DSV/10618775
