# Predicting Complaint Escalation for Mobile Subscribers

## Project Overview

This academic machine learning project explores whether a mobile
subscriber's complaint will be escalated (`Yes` or `No`). The problem is
framed around **Zenith Mobile Networks**, where the service team wants
to identify complaints that may require attention from senior agents
before they are escalated.

The model uses information from three sources:

-   Subscriber profile information
-   Call quality logs
-   Complaint ticket history

The project was completed for institutional learning and
experimentation. It is not presented as a production-ready escalation
system.

## Problem Statement

Build a binary classification workflow that predicts the `escalated`
status of a subscriber using profile, network call-quality and
complaint-ticket information.

-   **Target:** `escalated`
-   **Classes:** `Yes` and `No`
-   **Positive class of interest:** `Yes` (complaint escalated)

## Dataset Description

The notebook loads five CSV files:

  ----------------------------------------------------------------------------
  File                         Purpose               Rows reported in notebook
  ---------------------------- --------------------- -------------------------
  `subscriber_profile.csv`     Subscriber profile                        2,360
                               attributes            

  `call_quality_logs.csv`      Call quality                             18,920
                               measurements          

  `complaint_tickets.csv`      Complaint ticket                          5,918
                               records               

  `escalated_labeled.csv`      Known escalation                          2,006
                               labels for            
                               training/evaluation   

  `escalated_to_predict.csv`   Subscriber IDs                              354
                               requiring predictions 
  ----------------------------------------------------------------------------

The three feature datasets are joined using `subscriber_id`. The labeled
file contains 2,006 records: 1,380 `No` and 626 `Yes`.

> **Dataset note:** These are institution-provided study datasets. Their
> quality and representativeness may be limited. Confirm that
> redistribution is permitted before committing any dataset files to a
> public repository. If permission is not available, keep the data out
> of GitHub and document how an authorized reviewer can obtain it.

## Tools and Libraries

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   Jupyter Notebook / Google Colab
-   Joblib
-   Data profiling package used for an exploratory HTML report

## Project Workflow

### 1. Data loading and exploration

The notebook loads the three feature datasets, the labeled dataset and
the prediction-ID dataset. It examines dataset dimensions, data types,
descriptive statistics, duplicate records and missing values.

Exploratory analysis includes subscriber plan and region distributions,
tenure distribution, call drop-rate and latency distributions, complaint
issue types, and missing-value checks.

### 2. Data quality checks and cleaning

The notebook records the following data-quality observations and
treatments:

-   `segment`: 86 missing values in the 2,360-row subscriber profile
    (about 3.64%); handled later by categorical imputation.
-   `latency_ms`: 170 negative values were converted to absolute values
    after the notebook treated them as incorrect negative signs.
-   `days_open`: 146 missing values in the complaint-ticket dataset
    (about 2.47%); handled later through numerical imputation after
    aggregation.
-   No negative `drop_rate_pct` values or negative `days_open` values
    were reported in the notebook.

The absolute-value correction for latency is an assumption about the
source data. It should be validated with the institution or data
documentation where possible.

### 3. Feature aggregation

The call logs are aggregated by `subscriber_id` to create:

-   `total_calls`
-   `avg_drop_rate`
-   `max_drop_rate`
-   `avg_latency`
-   `max_latency`
-   `min_latency`

Complaint tickets are aggregated by `subscriber_id` to create:

-   `total_tickets`
-   `avg_days_open`
-   `max_days_open`
-   `min_days_open`
-   `unique_issue_types`

### 4. Data merging and target preparation

The aggregated call and ticket features are left-joined to the
subscriber profile. The labeled dataset is then inner-joined to form the
modeling dataset.

The notebook reports 374 subscribers without complaint-ticket records.
It fills `total_tickets` and `unique_issue_types` with zero. Other
missing numerical and categorical values are left for the preprocessing
pipelines.

The final labeled dataset reported by the notebook has 2,006 rows and 17
columns, including the target.

### 5. Train-test split

-   Training set: 1,604 rows
-   Test set: 402 rows
-   Split: 80:20
-   `random_state=42`
-   `stratify=y` to preserve the target-class proportions

The target distribution is:

  Class     Count
  ------- -------
  No        1,380
  Yes         626

### 6. Preprocessing

The notebook separates numerical and categorical columns and defines a
`ColumnTransformer`:

**Numerical pipeline** - `SimpleImputer(strategy="median")` -
`StandardScaler()`

**Categorical pipeline** - `SimpleImputer(strategy="most_frequent")` -
`OneHotEncoder(handle_unknown="ignore")`

The final hyperparameter-search workflow places preprocessing and
Logistic Regression together in a Scikit-learn pipeline. This allows
preprocessing to be fitted within each cross-validation training fold.

### 7. Model training and class imbalance

The classifier used in the documented final workflow is **Logistic
Regression**.

The notebook first evaluates a standard Logistic Regression model, then
explores `class_weight="balanced"` to give additional weight to the
minority `Yes` class.

### 8. Cross-validation and hyperparameter tuning

The notebook uses F1-score for the positive class (`Yes`) as the
GridSearchCV scoring metric.

The parameter grid is:

-   `C`: 0.01, 0.1, 1, 10, 100
-   `class_weight`: `balanced`
-   `solver`: `liblinear`, `lbfgs`

The best parameters reported in the notebook are:

``` python
{
    "classifier__C": 0.01,
    "classifier__class_weight": "balanced",
    "classifier__solver": "liblinear"
}
```

Best cross-validation F1-score reported: **0.5824**.

### 9. Model evaluation

The notebook reports the following results on the 402-row test set.

**Initial Logistic Regression**

  Metric                 Score
  ------------------- --------
  Accuracy              0.7363
  Precision (`Yes`)     0.6000
  Recall (`Yes`)        0.4560
  F1-score (`Yes`)      0.5182

**Tuned Logistic Regression (`class_weight="balanced"`)**

  Class                    Precision   Recall   F1-score   Support
  ---------------------- ----------- -------- ---------- ---------
  No                            0.86     0.69       0.76       277
  Yes                           0.52     0.75       0.61       125
  **Overall accuracy**                          **0.71**   **402**

The tuned model identifies a larger proportion of escalated complaints
than the initial model, with a trade-off in precision. The `Yes`-class
recall is 0.75, while its precision is 0.52.

### 10. Prediction and model export

For the 354 subscriber IDs in `escalated_to_predict.csv`, the notebook:

1.  Joins the IDs to the prepared subscriber feature table.
2.  Aligns the prediction features to the training feature columns.
3.  Uses the tuned pipeline to predict `Yes`/`No`.
4.  Calculates the escalation probability.
5.  Saves the results as `escalation_predictions.csv`.
6.  Saves the fitted pipeline as `escalation_model.pkl` using Joblib.

The prediction output contains:

-   `subscriber_id`
-   `escalated_prediction`
-   `escalation_probability`

The notebook shows a sample of the generated predictions, but the CSV
and model files should be generated by running the notebook; they are
not separate source datasets.

## Repository Structure

A suggested repository layout is:

``` text
ML-Classification-Project/
│
├── README.md
├── newclassification.ipynb
├── subscriber_profile.csv
├── call_quality_logs.csv
├── complaint_tickets.csv
├── escalated_labeled.csv
├── escalated_to_predict.csv
│
├── outputs/
│   └── escalation_predictions.csv
│
└── models/
    └── escalation_model.pkl
```

Only include data and generated artifacts that you are permitted to
publish. If the datasets are restricted, omit the CSV files and add
instructions for obtaining them. The notebook currently expects the CSV
files to be available in its working directory.

## How to Run

1.  Clone or download this repository.
2.  Place the permitted CSV files in the notebook's working directory,
    using the filenames referenced in the notebook.
3.  Open `newclassification.ipynb` in Jupyter Notebook or Google Colab.
4.  Install the required Python libraries if they are not already
    available.
5.  Run the notebook cells in order.
6.  The final cells create `escalation_predictions.csv` and
    `escalation_model.pkl`.

The notebook was developed with a Google Colab upload workflow. The
file-loading cells may need adjustment when running locally.

## Limitations and Further Improvements

-   The datasets are educational and may not fully represent real-world
    subscriber behaviour.
-   The labeled dataset is relatively small (2,006 records) and has a
    moderate class imbalance.
-   Converting negative latency values to absolute values relies on an
    assumption about data-entry errors.
-   Subscribers without tickets have missing ticket-duration aggregates;
    the notebook imputes these through the general numerical median
    rather than explicitly representing "no ticket" for those duration
    features.
-   The reported performance is based on one stratified hold-out split.
    Further validation on an independent dataset would be needed before
    real-world use.
-   Precision for the `Yes` class is 0.52, so some predicted escalations
    will be false positives.
-   The model's probability outputs should not be treated as calibrated
    risk estimates without a separate calibration assessment.

Potential next steps include validating data corrections with the data
owner, comparing additional classifiers, checking probability
calibration, assessing feature importance with a reproducible workflow,
and testing on new representative data.

## Academic Disclaimer

This project was developed for educational and study purposes using
institution-provided datasets. The implementation and reported metrics
describe the notebook's experimental workflow and should not be
interpreted as evidence of production readiness or real-world business
impact.

## Notebook Review Notes Before Publishing

A code review of the uploaded notebook found a few items worth fixing or
clearly documenting before presenting the notebook as fully
reproducible:

1.  **Cross-validation variable:** `cv` is used by the standalone
    `cross_val_score` call and by `GridSearchCV`, but no definition of
    `cv` appears in the uploaded notebook. Define it explicitly (for
    example, a 5-fold `StratifiedKFold` object with a fixed random seed)
    before those cells.
2.  **Feature-importance variable:** `importance_df` is used in the
    feature-coefficient plot, but its creation is not present in the
    uploaded notebook. Add the coefficient-to-feature-name extraction
    code or remove that plot cell until it is implemented.
3.  **Balanced-model evaluation cell:** The cell that trains
    `model_balanced` creates `y_pred_balanced`, but the following metric
    cell evaluates `y_pred` instead. Evaluate `y_pred_balanced` there if
    the intention is to report the balanced model's standalone metrics.
4.  **Confusion-matrix output consistency:** The stored confusion-matrix
    count output appears inconsistent with the tuned classification
    report. Re-run the notebook from a clean runtime, in order, and
    regenerate the confusion matrix and printed counts from the same
    final `best_model` predictions.
5.  **Cross-validation preprocessing:** The standalone cross-validation
    cell scores `model_balanced` on `X_train_processed`, where the
    preprocessor was fitted once on the full training split before the
    folds. That can leak information across CV folds. Prefer
    cross-validating a pipeline containing both `preprocessor` and the
    classifier. The `GridSearchCV` section already follows this safer
    pipeline pattern.
6.  **Prediction-ID matching:** Before prediction, check that every ID
    in `escalated_to_predict.csv` matched a row in the merged feature
    table. A left join can otherwise produce rows with entirely missing
    features without raising an error.

These notes do not invalidate the academic exercise, but addressing them
will make the notebook easier to reproduce and its reported metrics more
dependable.
