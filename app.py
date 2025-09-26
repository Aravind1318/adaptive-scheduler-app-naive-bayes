import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import joblib

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Adaptive Scheduling", layout="wide")
st.title("Adaptive Scheduling — AI + Adaptive Allocation (XGBoost)")

# -------------------------
# Upload dataset
# -------------------------
uploaded_file = st.file_uploader("Upload your scheduling dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV with your historical jobs (must include a machine_available column).")
    st.stop()

df = pd.read_csv(uploaded_file)

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.lower()

st.write("### Columns in dataset:", list(df.columns))
st.write("### Dataset preview", df.head())

# -------------------------
# Target and features
# -------------------------
if "machine_available" not in df.columns:
    st.error("❌ Could not find 'machine_available' column in your dataset. Please check column names.")
    st.stop()

target_col = "machine_available"
ignore_cols = ["job_id"] if "job_id" in df.columns else []
feature_cols = [c for c in df.columns if c != target_col and c not in ignore_cols]

st.write("Using features:", feature_cols)

X = df[feature_cols]
y = df[target_col]

# -------------------------
# Balance dataset
# -------------------------
df_balanced = df.copy()
if y.value_counts().min() / y.value_counts().max() < 0.5:
    st.warning("Dataset seems imbalanced, applying upsampling...")
    majority_class = y.value_counts().idxmax()
    df_majority = df[df[target_col] == majority_class]
    df_minority = df[df[target_col] != majority_class]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[feature_cols]
y = df_balanced[target_col]

# -------------------------
# Encode target
# -------------------------
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, "label_encoder.pkl")

# -------------------------
# Preprocessing
# -------------------------
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

# -------------------------
# Train-test split
# -------------------------
if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2:
    stratify = y
else:
    stratify = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# -------------------------
# Transform features
# -------------------------
X_train_dense = preprocessor.fit_transform(X_train)
X_test_dense = preprocessor.transform(X_test)

if hasattr(X_train_dense, "toarray"):
    X_train_dense = X_train_dense.toarray()
    X_test_dense = X_test_dense.toarray()

# -------------------------
# Train XGBoost (no tuning, fixed params)
# -------------------------
xgb = XGBClassifier(
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb.fit(X_train_dense, y_train)
joblib.dump((preprocessor, xgb), "scheduling_model.pkl")

# -------------------------
# Evaluate
# -------------------------
y_pred = xgb.predict(X_test_dense)
acc = accuracy_score(y_test, y_pred)

st.success("✅ Model trained (XGBoost).")
st.write(f"**Accuracy on test set:** {acc:.4f}")

# -------------------------
# Session state initialization
# -------------------------
machine_ids = sorted([str(x) for x in le.inverse_transform(np.unique(y))])

if "machine_loads" not in st.session_state:
    st.session_state.machine_loads = {m: 0 for m in machine_ids}
if "manpower_available" not in st.session_state:
    st.session_state.manpower_available = 100
if "assigned_tasks" not in st.session_state:
    st.session_state.assigned_tasks = {m: [] for m in machine_ids}

# -------------------------
# Real-time input form
# -------------------------
st.subheader("Real-time Task Scheduling")
with st.form("task_form"):
    inputs = {}
    for col in feature_cols:
        if col in categorical:
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) > 0:
                choice = st.selectbox(f"{col}", options=unique_vals, key=f"inp_{col}")
            else:
                choice = st.text_input(f"{col}", key=f"inp_{col}")
            inputs[col] = choice
        else:
            val = st.number_input(f"{col}", step=1.0, key=f"inp_{col}")
            inputs[col] = val
    submit = st.form_submit_button("Allocate Task")

# -------------------------
# Prediction & Allocation
# -------------------------
def build_input_row(inputs):
    return pd.DataFrame([inputs], columns=feature_cols)

if submit:
    inp_df = build_input_row(inputs)
    inp_dense = preprocessor.transform(inp_df).toarray()

    try:
        assigned_encoded = xgb.predict(inp_dense)[0]
        assigned = le.inverse_transform([assigned_encoded])[0]
        st.success(f"✅ Allocated to Machine {assigned}")

        est_col = [c for c in feature_cols if "estimated" in c]
        est_time = float(inputs.get(est_col[0], 1.0)) if est_col else 1.0

        manpower_col = [c for c in feature_cols if "manpower" in c]
        manpower_req = int(inputs.get(manpower_col[0], 1)) if manpower_col else 1

        if manpower_req > st.session_state.manpower_available:
            st.error(f"Not enough manpower: required {manpower_req}, available {st.session_state.manpower_available}")
        else:
            st.session_state.machine_loads[str(assigned)] += int(est_time)
            st.session_state.manpower_available -= manpower_req
            st.session_state.assigned_tasks[str(assigned)].append(str(inputs))

            st.info(f"Machine {assigned} load now {st.session_state.machine_loads[str(assigned)]} hrs. "
                    f"Manpower left: {st.session_state.manpower_available}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------
# Reset system
# -------------------------
if st.button("🔄 Reset System"):
    st.session_state.machine_loads = {m: 0 for m in machine_ids}
    st.session_state.manpower_available = 100
    st.session_state.assigned_tasks = {m: [] for m in machine_ids}
    st.success("System has been reset!")

# -------------------------
# Monitoring panel
# -------------------------
st.subheader("Current System Load")

safe_machine_loads = {str(k): int(v) for k, v in st.session_state.machine_loads.items()}
st.table(pd.DataFrame([
    {"Machine": k, "Load_hrs": v, "Assigned_Tasks": len(st.session_state.assigned_tasks.get(k, []))}
    for k, v in safe_machine_loads.items()
]))

loads_series = pd.Series({k: v for k, v in safe_machine_loads.items()})
st.bar_chart(loads_series.astype(float))

st.write("Available Manpower:", int(st.session_state.manpower_available))

for mkey in sorted(st.session_state.assigned_tasks.keys(), key=lambda x: x):
    with st.expander(f"Machine {mkey} tasks ({len(st.session_state.assigned_tasks[mkey])})"):
        for t in st.session_state.assigned_tasks[mkey]:
            st.write("-", t)
