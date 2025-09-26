import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Adaptive Scheduling", layout="wide")
st.title("Adaptive Scheduling — AI + Adaptive Allocation (Improved Random Forest Model)")

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
# Balance dataset across ALL classes
# -------------------------
df_balanced = pd.DataFrame()
max_size = y.value_counts().max()
for label in y.unique():
    df_class = df[df[target_col] == label]
    df_class_balanced = resample(
        df_class,
        replace=True,
        n_samples=max_size,
        random_state=42
    )
    df_balanced = pd.concat([df_balanced, df_class_balanced])

X = df_balanced[feature_cols]
y = df_balanced[target_col]

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Random Forest model
# -------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(X_train, y_train)
joblib.dump(model, "scheduling_model.pkl")

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success("✅ Model trained (Random Forest).")
st.write(f"**Accuracy on test set:** {acc:.4f}")

# -------------------------
# Feature Importance
# -------------------------
st.subheader("Feature Importance (Random Forest)")

try:
    # Extract trained RandomForest from pipeline
    rf = model.named_steps["classifier"]
    if hasattr(rf, "feature_importances_"):
        # Match transformed feature names
        ohe = model.named_steps["preprocessor"].named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(categorical) if categorical else []
        feature_names = np.concatenate([numerical, cat_features])
        importances = rf.feature_importances_

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

        st.dataframe(fi_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        fi_df.plot(kind="barh", x="Feature", y="Importance", ax=ax, legend=False)
        plt.gca().invert_yaxis()
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not display feature importance: {e}")

# -------------------------
# Session state initialization
# -------------------------
machine_ids = sorted([str(x) for x in y.unique()])

if "machine_loads" not in st.session_state:
    st.session_state.machine_loads = {m: 0 for m in machine_ids}
if "manpower_available" not in st.session_state:
    st.session_state.manpower_available = 100
if "assigned_tasks" not in st.session_state:
    st.session_state.assigned_tasks = {m: [] for m in machine_ids}

# -------------------------
# Real-time input form (with dropdowns for categorical features)
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
# Transform input row
# -------------------------
def build_input_row(inputs):
    return pd.DataFrame([inputs], columns=feature_cols)

# -------------------------
# Allocation logic
# -------------------------
if submit:
    inp_df = build_input_row(inputs)
    try:
        assigned = model.predict(inp_df)[0]
        st.success(f"✅ Allocated to Machine {assigned}")

        est_col = [c for c in feature_cols if "estimated" in c][0] if any("estimated" in c for c in feature_cols) else None
        est_time = float(inputs.get(est_col, 1.0))

        manpower_col = [c for c in feature_cols if "manpower" in c][0] if any("manpower" in c for c in feature_cols) else None
        manpower_req = int(inputs.get(manpower_col, 1))

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
