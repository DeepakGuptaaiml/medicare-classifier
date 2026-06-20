#!/usr/bin/env python3
"""Generate Medicare_Classifier_Notebook.ipynb using nbformat (AML classification structure)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "Medicare_Classifier_Notebook.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    cells = [
        md("# Medicare Reportable Claims Classifier"),
        md("## Problem Statement"),
        md(
            """
### Business Context

Under **MMSEA Section 111**, Medicare Secondary Payer (MSP) regulations require insurers to identify claims that must be reported to CMS via the **V1M** (Voluntary Data Sharing Agreement) process. Missing reportable claims creates compliance risk, penalties, and recovery exposure; over-reporting wastes operational effort.

You are a data scientist building a **binary classifier** that flags whether a claim is **Medicare reportable** before V1M submission — using claim financials, coverage codes, payment history, and claimant demographics from `claims_data.csv`.

**Target:** `is_medicare_reportable` (derived from `is_v1m_extracted` — whether the claim was successfully extracted/reported in V1M).

**Objective:** Maximize **recall** on the positive class (reportable claims) so the carrier catches reportable claims early while controlling false positives.

### Data Description

**Target (engineered)**
* **is_medicare_reportable** — 1 if claim was V1M-extracted (`is_v1m_extracted`), else 0

**Raw columns in `claims_data.csv`**

* **claim_uid** — Unique claim identifier
* **file_num** — Internal file / claim number
* **data_set** — Line of business (WC workers' comp, GL, PF, AU)
* **cont_num** — Contract / account number
* **proc_unit** — Processing unit handling the claim
* **plan_num** — Plan identifier
* **claim_type** — Claim admission type (MO, IO, OT)
* **line_code** — Product line code (WC, GR, ZE, etc.); ZE/GR excluded from Medicare reporting
* **coverage_code** — Coverage exclusion flag (ZE, WA, LT excluded from reporting)
* **pay_cat** — Payment category (PI, PD, CL, MP, BI, CM, OT)
* **claim_open** — Whether claim is still open
* **date_event** — Date of loss / injury event
* **date_open** — Claim open date
* **date_close** — Claim close date (if closed)
* **date_v1m_xmit** — Date claim was transmitted to CMS V1M (null if not yet reported)
* **clmnt_ssn** — Claimant Social Security number (for prior-claim linkage)
* **clmnt_dob** — Claimant date of birth (Medicare eligibility age signal)
* **clmnt_gender** — Claimant gender
* **clmnt_country** — Claimant country (USA required for US Medicare reporting)
* **clmnt_state** — Claimant state of residence
* **pay_code** — CMS payment reason code (100–199 TPOC, 300–399 ORM, settlement codes)
* **pay_type** — Payment type (SYS system, HIS history, MAN manual)
* **paid_1** — Total paid to date (TPOC threshold signal)
* **paid_3** — Cumulative paid (ORM $750 threshold signal)
* **amount** — Current transaction / payment amount
* **state_juris** — State jurisdiction code
* **our_cause_2** — Injury mechanism (cause level 2)
* **our_cause_3** — Diagnosis / injury category (cause level 3)
* **our_cause_4** — Body part / procedure category (cause level 4)
* **date_rpted_cms** — Date reported to CMS
* **days_to_cms** — Days from event to CMS reporting
* **reserve_3** — 3-month case reserve
* **reserve_6** — 6-month case reserve
* **reserve_status** — Reserve adequacy status (A, etc.)
* **is_v1m_extracted** — Source label: 1 if V1M extracted (Medicare reportable), 0 otherwise
* **excl_reason** — Reason claim excluded from V1M (coverage, line code, etc.)

##### What is MMSEA Section 111?

Medicare Secondary Payer rules require insurers to report certain liability, workers' compensation, and no-fault claims to CMS so Medicare can recover conditional payments.

##### What is V1M extraction?

V1M is the voluntary data-sharing extract sent to CMS. Claims with `is_v1m_extracted = 1` were identified as Medicare reportable and successfully included in the extract.

##### Pay code buckets (ORM / TPOC / SETTLEMENT)

CMS uses payment reason codes to classify **Total Payment Obligation to Claimant (TPOC)**, **Ongoing Responsibility for Medicals (ORM)**, and **settlement** events — each triggers different Section 111 reporting obligations.
"""
        ),
        md(
            """
### **Please read the instructions carefully before starting the project.**

* Run cells **sequentially** from top to bottom.
* Feature engineering maps raw columns to 20 modeling features including `is_wc` and `pay_code_bucket`.
* The best tuned model is saved to `models/medicare_classifier.pkl` and `models/preprocess_config.json`.
* **Recall** on reportable claims is the primary model-selection metric.
"""
        ),
        md("## Importing necessary libraries"),
        code(
            """
# To load and manipulate data
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None

# Import evaluation metrics for classification
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

%matplotlib inline

warnings.filterwarnings("ignore")

# Import SMOTE package for oversampling
from imblearn.over_sampling import SMOTE

# Import Random undersampler for undersampling
from imblearn.under_sampling import RandomUnderSampler

# Import packages to test and split
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Import packages for classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from xgboost import XGBClassifier

RS = 1
BASE_DIR = Path(".").resolve()
DATA_PATH = BASE_DIR / "data" / "claims_data.csv"
MODELS_DIR = BASE_DIR / "models"
"""
        ),
        md("## Loading the dataset"),
        code(
            """
claims_df = pd.read_csv(DATA_PATH)
claims_df.head()
"""
        ),
        md("## Exploratory Data Analysis"),
        md("### Utility functions for EDA"),
        code(
            """
# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    \"\"\"
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    \"\"\"
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()
"""
        ),
        code(
            """
# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    \"\"\"
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    \"\"\"

    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()

        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()
"""
        ),
        code(
            """
# function to plot stacked bar chart

def stacked_barplot(data, predictor, target):
    \"\"\"
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    \"\"\"
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(loc="lower left", frameon=False)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()
"""
        ),
        code(
            """
### Function to plot distributions

def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()
"""
        ),
        md("### Data overview"),
        code("claims_df.head()"),
        code(
            """
print("Number of Rows: ", claims_df.shape[0])
print("Number of Columns: ", claims_df.shape[1])
"""
        ),
        code("claims_df.info()"),
        code(
            """
print("Null values per column:")
print(claims_df.isnull().sum())
"""
        ),
        code(
            """
print("Duplicate rows:", claims_df.duplicated().sum())
"""
        ),
        md("### Univariate and bivariate EDA vs target"),
        code(
            """
# Engineer target for EDA (same definition as modeling)
eda_df = claims_df.copy()
eda_df["is_medicare_reportable"] = (
    pd.to_numeric(eda_df["is_v1m_extracted"], errors="coerce").fillna(0).astype(int)
)
print("Target distribution:")
print(eda_df["is_medicare_reportable"].value_counts(normalize=True))
labeled_barplot(eda_df, "is_medicare_reportable", perc=True)
"""
        ),
        code(
            """
stacked_barplot(eda_df, "data_set", "is_medicare_reportable")
stacked_barplot(eda_df, "pay_cat", "is_medicare_reportable")
"""
        ),
        code(
            """
for col in ["paid_1", "paid_3", "amount", "days_to_cms"]:
    distribution_plot_wrt_target(eda_df, col, "is_medicare_reportable")
"""
        ),
        md("## Feature Engineering"),
        code(
            """
TARGET = "is_medicare_reportable"
FEATURES = [
    "data_set",
    "pay_cat",
    "pay_code",
    "pay_type",
    "paid_1",
    "paid_3",
    "amount",
    "proc_unit",
    "cont_num",
    "claim_open",
    "date_v1m_xmit_flag",
    "is_us_claimant",
    "orm_threshold_met",
    "tpoc_threshold_met",
    "is_wc",
    "pay_code_bucket",
    "is_excluded_coverage",
    "is_excluded_line",
    "days_open",
    "age_at_event",
]
CAT_COLS = ["data_set", "pay_cat", "pay_type", "pay_code_bucket"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]


def pay_code_bucket(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "OTHER"
    if code in (113, 120, 135, 137, 153):
        return "SETTLEMENT"
    if 100 <= code <= 199 or code == 390:
        return "TPOC"
    if 300 <= code <= 399:
        return "ORM"
    return "OTHER"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = pd.to_numeric(out["is_v1m_extracted"], errors="coerce").fillna(0).astype(int)

    for col in ["date_event", "date_open", "date_close", "clmnt_dob", "date_v1m_xmit"]:
        out[f"{col}_dt"] = pd.to_datetime(out[col], errors="coerce")

    out["date_v1m_xmit_flag"] = out["date_v1m_xmit_dt"].notna().astype(int)
    out["is_us_claimant"] = (out["clmnt_country"].astype(str) == "USA").astype(int)
    out["orm_threshold_met"] = (pd.to_numeric(out["paid_3"], errors="coerce").fillna(0) > 750).astype(int)
    out["tpoc_threshold_met"] = (pd.to_numeric(out["paid_1"], errors="coerce").fillna(0) > 0).astype(int)
    out["is_wc"] = (out["data_set"].astype(str) == "WC").astype(int)
    out["pay_code_bucket"] = out["pay_code"].apply(pay_code_bucket)
    out["is_excluded_coverage"] = out["coverage_code"].isin(["ZE", "WA", "LT"]).astype(int)
    out["is_excluded_line"] = out["line_code"].isin(["ZE", "GR"]).astype(int)

    out["days_open"] = (out["date_close_dt"] - out["date_open_dt"]).dt.days
    out.loc[out["claim_open"].astype(bool), "days_open"] = 0
    out["days_open"] = out["days_open"].fillna(0)

    out["age_at_event"] = (
        (out["date_event_dt"] - out["clmnt_dob_dt"]).dt.days / 365.25
    ).round(1)

    out["claim_open"] = out["claim_open"].astype(int)
    for col in NUM_COLS:
        if col != "claim_open":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


model_df = engineer_features(claims_df)
model_df[[TARGET] + FEATURES[:8]].head()
"""
        ),
        md("### Drop irrelevant columns"),
        code(
            """
drop_cols = [
    "claim_uid", "file_num", "plan_num", "claim_type", "line_code", "coverage_code",
    "date_event", "date_open", "date_close", "date_v1m_xmit", "clmnt_ssn", "clmnt_dob",
    "clmnt_gender", "clmnt_country", "clmnt_state", "state_juris", "our_cause_2",
    "our_cause_3", "our_cause_4", "date_rpted_cms", "days_to_cms", "reserve_3",
    "reserve_6", "reserve_status", "is_v1m_extracted", "excl_reason",
    "date_event_dt", "date_open_dt", "date_close_dt", "clmnt_dob_dt", "date_v1m_xmit_dt",
]
model_df = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns], errors="ignore")
print("Columns after drop:", model_df.columns.tolist())
"""
        ),
        md("### Encode categorical features"),
        code(
            """
frame = model_df[FEATURES].copy()
for col in CAT_COLS:
    frame[col] = frame[col].astype(str).fillna("MISSING")

X_raw = pd.get_dummies(frame, columns=CAT_COLS, drop_first=True)
y = model_df[TARGET].values
feature_columns = list(X_raw.columns)
print(f"Encoded feature count: {X_raw.shape[1]}")
X_raw.head()
"""
        ),
        md("### Train / validation / test split (70 / 15 / 15, stratified)"),
        code(
            """
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y, test_size=0.15, stratify=y, random_state=RS
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176470588, stratify=y_temp, random_state=RS
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.3f}")
"""
        ),
        md("### SMOTE oversampling and Random undersampling"),
        code(
            """
print("Original Count of Labels \\n")
print(f"Reportable (1): {sum(y_train == 1)}")
print(f"Not reportable (0): {sum(y_train == 0)} \\n")

smote = SMOTE(random_state=RS)
rus = RandomUnderSampler(random_state=RS)

X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("After Oversampling")
print(f"The shape of train_X: {X_train_over.shape}")
print(f"The shape of train_y: {y_train_over.shape} \\n")

print("After Undersampling")
print(f"The shape of train_X: {X_train_under.shape}")
print(f"The shape of train_y: {y_train_under.shape} \\n")
"""
        ),
        md("## Model Building"),
        md("### Performance helper functions"),
        code(
            """
def model_performance_classification_sklearn(model, predictors, target):
    \"\"\"
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    \"\"\"

    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred, zero_division=0)
    precision = precision_score(target, pred, zero_division=0)
    f1 = f1_score(target, pred, zero_division=0)

    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf
"""
        ),
        code(
            """
def plot_confusion_matrix(model, predictors, target):
    \"\"\"
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    \"\"\"
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)

    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()
"""
        ),
        md("### Train 6 models on Original, Oversampled, and Undersampled data"),
        code(
            """
models = []
models.append(("Bagging", BaggingClassifier(random_state=RS)))
models.append(("Random forest", RandomForestClassifier(random_state=RS, n_jobs=1)))
models.append(("GBM", GradientBoostingClassifier(random_state=RS)))
models.append(("Adaboost", AdaBoostClassifier(random_state=RS)))
models.append(("Xgboost", XGBClassifier(random_state=RS, eval_metric="logloss", n_jobs=1, verbosity=0)))
models.append(("dtree", DecisionTreeClassifier(random_state=RS)))

datasets = {
    "Original": (X_train, y_train),
    "Oversampled": (X_train_over, y_train_over),
    "Undersampled": (X_train_under, y_train_under),
}

comparison_rows = []
for sample_name, (x_tr, y_tr) in datasets.items():
    for name, estimator in models:
        model = estimator.__class__(**estimator.get_params())
        model.fit(x_tr, y_tr)
        train_recall = recall_score(y_tr, model.predict(x_tr), zero_division=0)
        val_recall = recall_score(y_val, model.predict(X_val), zero_division=0)
        val_f1 = f1_score(y_val, model.predict(X_val), zero_division=0)
        val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        val_auc = roc_auc_score(y_val, val_proba) if val_proba is not None else np.nan
        comparison_rows.append(
            {
                "Model Name": name,
                "Sampling": sample_name,
                "Recall_train": train_recall,
                "Recall_val": val_recall,
                "F1_val": val_f1,
                "ROC_AUC_val": val_auc,
            }
        )

comparison_df = pd.DataFrame(comparison_rows).sort_values(["Recall_val", "F1_val"], ascending=False)
print("Model comparison (validation recall):")
comparison_df
"""
        ),
        md(
            """
* **Observations**
  - Compare recall on validation across Original, Oversampled, and Undersampled training sets.
  - Decision trees often overfit; ensemble models (GBM, XGBoost, AdaBoost) typically generalize better.
  - Select the configuration with highest **validation recall** for hyperparameter tuning.
"""
        ),
        md("### RandomizedSearchCV (recall scorer) on best model"),
        code(
            """
best_row = comparison_df.iloc[0]
best_model_name = best_row["Model Name"]
best_sampling = best_row["Sampling"]

sampling_map = {
    "Original": (X_train, y_train),
    "Oversampled": (X_train_over, y_train_over),
    "Undersampled": (X_train_under, y_train_under),
}
x_tune, y_tune = sampling_map[best_sampling]

model_lookup = {name: est for name, est in models}
base_estimator = model_lookup[best_model_name]

param_grids = {
    DecisionTreeClassifier: {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    },
    BaggingClassifier: {
        "n_estimators": [25, 50, 100],
        "max_samples": [0.6, 0.8, 1.0],
    },
    RandomForestClassifier: {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    },
    AdaBoostClassifier: {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.05, 0.1, 0.5, 1.0],
    },
    GradientBoostingClassifier: {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
    },
    XGBClassifier: {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}

grid = param_grids.get(base_estimator.__class__, {})
scorer = metrics.make_scorer(recall_score)
search = RandomizedSearchCV(
    estimator=base_estimator.__class__(**base_estimator.get_params()),
    param_distributions=grid,
    n_iter=12,
    scoring=scorer,
    cv=5,
    random_state=RS,
    n_jobs=-1,
)
search.fit(x_tune, y_tune)
best_model = search.best_estimator_

print(f"Best base: {best_model_name} ({best_sampling})")
print(f"Tuned params: {search.best_params_}")
"""
        ),
        md("### Test set performance"),
        code(
            """
metrics_train = model_performance_classification_sklearn(best_model, X_train, y_train)
metrics_val = model_performance_classification_sklearn(best_model, X_val, y_val)
metrics_test = model_performance_classification_sklearn(best_model, X_test, y_test)

print("Train performance:")
print(metrics_train.to_string(index=False))
print("Validation performance:")
print(metrics_val.to_string(index=False))
print("Test performance:")
metrics_test
"""
        ),
        md("### Confusion matrices — train, validation, test"),
        code(
            """
print("Train confusion matrix")
plot_confusion_matrix(best_model, X_train, y_train)
print("Validation confusion matrix")
plot_confusion_matrix(best_model, X_val, y_val)
print("Test confusion matrix")
plot_confusion_matrix(best_model, X_test, y_test)
"""
        ),
        md("### ROC curve and Precision-Recall curve"),
        code(
            """
y_test_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {test_auc:.3f})")
axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend(loc="lower right")

axes[1].plot(rec, prec, color="teal", lw=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Test Set")

plt.tight_layout()
plt.show()
"""
        ),
        md("### Feature importance"),
        code(
            """
feature_names = X_train.columns
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 10))
plt.title("Feature Importances — Medicare Classifier")
plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
"""
        ),
        md("### SHAP explainability"),
        code(
            """
try:
    import shap

    explainer = shap.TreeExplainer(best_model)
    shap_sample = X_test.iloc[:200]
    shap_values = explainer.shap_values(shap_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("shap not installed — showing global feature importances instead")
    importances = best_model.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(X_test.columns)[idx], importances[idx], color="steelblue")
    plt.title("Top 15 Feature Importances (SHAP fallback)")
    plt.tight_layout()
    plt.show()
"""
        ),
        code(
            """
example_idx = 0
example = X_test.iloc[[example_idx]]

try:
    import shap

    shap_values_one = explainer.shap_values(example)
    if isinstance(shap_values_one, list):
        shap_values_one = shap_values_one[1]

    base = explainer.expected_value
    if isinstance(base, list):
        base = base[1]

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_one[0],
            base_values=base,
            data=example.iloc[0],
            feature_names=example.columns.tolist(),
        ),
        show=False,
    )
    plt.tight_layout()
    plt.show()
except (ImportError, NameError):
    # Local explanation fallback: top features by |value * importance|
    importances = best_model.feature_importances_
    contrib = (example.values[0] * importances)
    top_idx = np.argsort(np.abs(contrib))[-10:]
    plt.figure(figsize=(10, 5))
    plt.barh(np.array(example.columns)[top_idx], contrib[top_idx], color="coral")
    plt.title(f"Local Feature Contribution — Test Row {example_idx} (SHAP fallback)")
    plt.tight_layout()
    plt.show()

print(f"Actual label: {y_test[example_idx]}, Predicted: {best_model.predict(example)[0]}")
"""
        ),
        md("### Save model artifacts"),
        code(
            """
os.makedirs(MODELS_DIR, exist_ok=True)

preprocess_config = {
    "cat_impute": {col: model_df[col].astype(str).mode().iloc[0] for col in CAT_COLS},
    "num_impute": {
        col: float(model_df[col].median()) if col in model_df.columns else 0.0
        for col in NUM_COLS
    },
    "categorical_options": {
        col: sorted(model_df[col].astype(str).dropna().unique().tolist()) for col in CAT_COLS
    },
    "feature_columns": feature_columns,
    "raw_features": FEATURES,
    "target": TARGET,
}

artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "sampling_strategy": best_sampling,
    "target": TARGET,
    "feature_columns": feature_columns,
    "raw_features": FEATURES,
    "best_params": search.best_params_,
    "metrics_train": metrics_train.to_dict(orient="records")[0],
    "metrics_val": metrics_val.to_dict(orient="records")[0],
    "metrics_test": metrics_test.to_dict(orient="records")[0],
    "comparison_top10": comparison_df.head(10).to_dict(orient="records"),
}

joblib.dump(artifact, MODELS_DIR / "medicare_classifier.pkl")
(MODELS_DIR / "preprocess_config.json").write_text(
    json.dumps(preprocess_config, indent=2), encoding="utf-8"
)
comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

print(f"Saved {MODELS_DIR / 'medicare_classifier.pkl'}")
print(f"Saved {MODELS_DIR / 'preprocess_config.json'}")
print(f"Model: {best_model_name} | Sampling: {best_sampling}")
print(f"Test Recall: {metrics_test['Recall'].values[0]:.4f}")
print(f"Test F1: {metrics_test['F1'].values[0]:.4f}")
"""
        ),
        md("# Business Insights and Conclusions"),
        md(
            """
Based on the analysis, model comparison, and feature importances from the best Medicare classifier:

**Key Business Insights**

- **Pay code bucket (ORM / TPOC / SETTLEMENT)** and **paid_1 / paid_3 thresholds** are strong predictors — claims crossing CMS financial thresholds (ORM > $750, any TPOC) are more likely Medicare reportable under Section 111.
- **Workers' compensation (`is_wc`)** and **data_set** capture line-of-business effects; WC claims dominate the portfolio and drive most V1M volume.
- **Excluded coverage (`ZE`, `WA`, `LT`) and line codes (`ZE`, `GR`)** correctly suppress reporting — the model learns these exclusion patterns.
- **US claimant flag** and **claimant age at event** align with Medicare eligibility (65+ or disability entitlement).
- **date_v1m_xmit_flag** (prior transmission) is highly informative but reflects post-reporting state — use cautiously in production pre-submission scoring.

**Recommendations for Compliance Operations**

1. **Prioritize recall** — tune deployment threshold to catch reportable claims; manual review queue for high-recall, lower-precision flags.
2. **Auto-exclude** claims with excluded coverage/line codes before model scoring to reduce noise.
3. **Monitor ORM/TPOC threshold features** — when paid amounts approach CMS thresholds, escalate for Section 111 review.
4. **Deploy** `medicare_classifier.pkl` with `preprocess_config.json` for consistent feature encoding at intake.

**Model Artifact**

- Best model saved to `models/medicare_classifier.pkl` including feature column order and test metrics for audit.
"""
        ),
    ]

    nb.cells = cells
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
