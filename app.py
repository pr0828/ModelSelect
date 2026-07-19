# ==========================================================
# ModelSelect
# Intelligent Machine Learning Model Recommendation System
# Fully dynamic: works with ANY uploaded CSV dataset
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ----------------------------------------------------------
# Page Settings
# ----------------------------------------------------------

st.set_page_config(
    page_title="ModelSelect",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ModelSelect")
st.subheader("Intelligent Machine Learning Model Recommendation System")
st.caption("Upload any CSV — target detection, preprocessing, model choice and evaluation all adapt automatically.")

st.markdown("---")

# ----------------------------------------------------------
# Upload Dataset
# ----------------------------------------------------------

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Uploaded Successfully")
    except Exception as e:
        st.error(f"Could not read this CSV file: {e}")
        st.stop()
else:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["Species"] = iris.target
    df["Species"] = df["Species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    st.info("No file uploaded — showing the built-in Iris dataset as a demo")

if df.shape[0] < 10 or df.shape[1] < 2:
    st.error("Dataset is too small to train on. Please upload a CSV with at least a couple of columns and 10+ rows.")
    st.stop()

# ----------------------------------------------------------
# Dataset Preview
# ----------------------------------------------------------

st.header("Dataset Preview")
st.dataframe(df, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])

st.markdown("---")

# ----------------------------------------------------------
# Target Column Selection
# ----------------------------------------------------------

st.header("🎯 Select Target Column")

columns = list(df.columns)

target_column = st.selectbox(
    "Select the column you want to predict",
    columns,
    index=len(columns) - 1,
    help="Defaults to the last column — change it if that isn't what you want to predict."
)

X_full = df.drop(columns=[target_column])
y_raw = df[target_column]

# Drop rows with a missing target — a model can't learn from an unknown label
missing_target_mask = y_raw.isna()
if missing_target_mask.any():
    st.warning(f"Dropping {int(missing_target_mask.sum())} row(s) with a missing target value")
    X_full = X_full[~missing_target_mask]
    y_raw = y_raw[~missing_target_mask]

# ----------------------------------------------------------
# Auto-detect Task Type (Classification vs Regression)
# ----------------------------------------------------------


def detect_task_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique(dropna=True)
        looks_discrete = (y.dropna() % 1 == 0).all()
        if n_unique <= 20 and looks_discrete:
            return "classification"
        return "regression"
    return "classification"


detected_task_type = detect_task_type(y_raw)

st.info(
    f"Detected task type: **{detected_task_type.title()}** "
    f"(target `{target_column}` has {y_raw.nunique()} unique value(s))"
)

if st.checkbox("Override detected task type"):
    task_type = st.radio(
        "Task Type",
        ["classification", "regression"],
        index=0 if detected_task_type == "classification" else 1,
        horizontal=True
    )
else:
    task_type = detected_task_type

st.markdown("---")

# ----------------------------------------------------------
# Automatic Feature Cleanup (generic — no hardcoded column names)
# ----------------------------------------------------------

st.header("🧹 Preprocessing")

n_rows = len(X_full)
auto_dropped = []

for col in X_full.columns:
    lower = str(col).lower().strip()
    name_looks_like_id = lower == "id" or lower.endswith("_id") or lower.endswith(" id")
    nunique = X_full[col].nunique(dropna=True)
    near_unique = n_rows > 20 and nunique >= 0.95 * n_rows
    if name_looks_like_id or near_unique:
        auto_dropped.append(col)

if auto_dropped:
    st.write("Automatically excluding likely identifier / free-text columns:")
    st.write(", ".join(f"`{c}`" for c in auto_dropped))
    X_full = X_full.drop(columns=auto_dropped)

remaining_columns = X_full.columns.tolist()
manual_drop = st.multiselect(
    "Optionally exclude additional columns from the features",
    remaining_columns
)
if manual_drop:
    X_full = X_full.drop(columns=manual_drop)

if X_full.shape[1] == 0:
    st.error("No feature columns remain after exclusions. Please keep at least one feature column.")
    st.stop()

numeric_cols = X_full.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_full.select_dtypes(exclude=np.number).columns.tolist()

st.write(f"Numeric features ({len(numeric_cols)}):", ", ".join(numeric_cols) if numeric_cols else "None")
st.write(f"Categorical features ({len(categorical_cols)}):", ", ".join(categorical_cols) if categorical_cols else "None")

missing_counts = X_full.isnull().sum()
if missing_counts.sum() == 0:
    st.success("No missing values in the selected features")
else:
    st.warning("Missing values found — will be imputed automatically (mean for numeric, most frequent for categorical)")
    st.dataframe(missing_counts[missing_counts > 0].rename("Missing Count"))

st.markdown("---")

# ----------------------------------------------------------
# Build a reusable preprocessing + modeling pipeline
# ----------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_pipeline, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_pipeline, categorical_cols))

preprocessor = ColumnTransformer(transformers)

# ----------------------------------------------------------
# Encode Target
# ----------------------------------------------------------

target_encoder = None

if task_type == "classification":
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw.astype(str))
else:
    y = pd.to_numeric(y_raw, errors="coerce")
    if y.isna().any():
        st.error("Target column contains values that can't be treated as numeric for regression. Try selecting a different target or overriding the task type.")
        st.stop()

X = X_full

# ----------------------------------------------------------
# Train / Test Split
# ----------------------------------------------------------

try:
    stratify_arg = y if task_type == "classification" and pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify_arg
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

st.success(f"Train/Test split completed — {len(X_train)} training rows, {len(X_test)} test rows")

st.markdown("---")

# ==========================================================
# MODEL TRAINING
# ==========================================================

st.header("🤖 Model Training")

if task_type == "classification":
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(),
        "K Nearest Neighbors": KNeighborsClassifier(),
    }
else:
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Machine": SVR(),
        "K Nearest Neighbors": KNeighborsRegressor(),
    }

results = []
trained_pipelines = {}

with st.spinner("Training candidate models..."):
    for name, model in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
        except Exception as e:
            st.warning(f"Skipping {name} — training failed ({e})")
            continue

        trained_pipelines[name] = pipe

        if task_type == "classification":
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "Precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
                "Recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
                "F1 Score": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4),
            })
        else:
            rmse = mean_squared_error(y_test, pred) ** 0.5
            results.append({
                "Model": name,
                "R2 Score": round(r2_score(y_test, pred), 4),
                "MAE": round(mean_absolute_error(y_test, pred), 4),
                "RMSE": round(rmse, 4),
            })

if not results:
    st.error("No model could be trained successfully on this dataset.")
    st.stop()

results_df = pd.DataFrame(results)
primary_metric = "Accuracy" if task_type == "classification" else "R2 Score"
results_df = results_df.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
best_pred = best_pipeline.predict(X_test)

# ==========================================================
# RESULTS TABLE
# ==========================================================

st.header("📊 Model Comparison")
st.dataframe(results_df, use_container_width=True)

# ==========================================================
# EVALUATION VISUAL (Confusion Matrix or Predicted vs Actual)
# ==========================================================

st.header("📊 Best Model Evaluation")

if task_type == "classification":
    cm = confusion_matrix(y_test, best_pred)
    labels = target_encoder.classes_ if target_encoder is not None else None

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    plt.colorbar(image)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_model_name}")

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, best_pred, alpha=0.6)
    min_val = min(min(y_test), min(best_pred))
    max_val = max(max(y_test), max(best_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual — {best_model_name}")
    st.pyplot(fig)

# ==========================================================
# METRICS (dynamic — driven by whatever columns are in results_df)
# ==========================================================

st.header("📈 Best Model Metrics")

metric_cols = [c for c in results_df.columns if c != "Model"]
metric_cards = st.columns(len(metric_cols))

for card, metric in zip(metric_cards, metric_cols):
    value = results_df.iloc[0][metric]
    if task_type == "classification":
        card.metric(metric, f"{value * 100:.2f}%")
    else:
        card.metric(metric, f"{value:.4f}")

# ==========================================================
# METRIC COMPARISON CHARTS (generic loop, no hardcoded metric names)
# ==========================================================

st.header("📈 Metric Comparison Across Models")

for metric in metric_cols:
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(results_df["Model"], results_df[metric])
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison")
    plt.xticks(rotation=20)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)

# ==========================================================
# PREDICTIONS TABLE
# ==========================================================

st.header("🔮 Predictions on Test Data")

if task_type == "classification" and target_encoder is not None:
    display_actual = target_encoder.inverse_transform(y_test)
    display_pred = target_encoder.inverse_transform(best_pred)
else:
    display_actual = y_test
    display_pred = best_pred

pred_df = X_test.copy().reset_index(drop=True)
pred_df["Actual"] = np.array(display_actual)
pred_df["Predicted"] = np.array(display_pred)

st.dataframe(pred_df, use_container_width=True)

# ==========================================================
# BEST MODEL SUMMARY
# ==========================================================

st.header("🏆 Recommended Model")

if task_type == "classification":
    st.success(f"""
Best Model : **{best_model_name}**

Accuracy : **{results_df.iloc[0]['Accuracy'] * 100:.2f}%**
""")
else:
    st.success(f"""
Best Model : **{best_model_name}**

R² Score : **{results_df.iloc[0]['R2 Score']:.4f}**
""")

# ==========================================================
# DOWNLOADS
# ==========================================================

st.header("📥 Downloads")

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        label="📥 Download Model Comparison CSV",
        data=results_df.to_csv(index=False),
        file_name="model_results.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        label="📥 Download Predictions CSV",
        data=pred_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

model_bytes = joblib.dump(best_pipeline, "best_model.pkl")
with open("best_model.pkl", "rb") as f:
    st.download_button(
        label="📥 Download Best Model (.pkl)",
        data=f,
        file_name="best_model.pkl",
        mime="application/octet-stream"
    )

st.caption("The downloaded model is a full pipeline (preprocessing + trained model), so it can be reused directly on new raw data with the same input columns.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
### 🤖 ModelSelect

Developed using

- Streamlit
- Scikit-Learn
- Pandas
- Matplotlib

Supports:

✅ Any uploaded CSV dataset

✅ Automatic target column detection

✅ Automatic classification vs regression detection

✅ Automatic preprocessing (missing values, encoding, scaling)

✅ Multi-model training & comparison

✅ Performance evaluation with task-appropriate metrics

✅ Predictions preview & model download
"""
)