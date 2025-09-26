import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Adaptive Scheduling", layout="wide")
st.title("Adaptive Scheduling — AI + Adaptive Allocation (Improved RF)")

# -------------------------
# Upload dataset
# -------------------------
uploaded_file = st.file_uploader("Upload your scheduling dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV with your historical jobs (must include Machine_Availability column).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Dataset preview", df.head())

# -------------------------
# Target & features
# -------------------------
target_col = "Machine_Availability"
ignore_cols = ["Job_ID"] if "Job_ID" in df.columns else []
feature_cols = [c for c in df.columns if c != target_col and c not in ignore_cols]
st.write("Using features:", feature_cols)

X = df[feature_cols]
y = df[target_col]

# -------------------------
# Preprocessing (OneHotEncoder for categorical features)
# -------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Model pipeline (Preprocessing + RandomForest)
# -------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
joblib.dump(model, "scheduling_model_rf.pkl")

# -------------------------
# Accuracy
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success("Model trained with OneHot + RandomForest.")
st.write(f"**Accuracy on test set:** {acc:.4f}")

# -------------------------
# Session state initialization
# -------------------------
machine_ids = sorted(y.unique())
if "machine_loads" not in st.session_state:
    st.session_state.machine_loads = {str(m): 0 for m in machine_ids}
if "manpower_available" not in st.session_state:
    st.session_state.manpower_available = 100
if "assigned_tasks" not in st.session_state:
    st.session_state.assigned_tasks = {str(m): [] for m in machine_ids}

# -------------------------
# Real-time input
# -------------------------
st.subheader("Real-time Task Scheduling")
with st.form("task_form"):
    inputs = {}
    for col in feature_cols:
        if col in categorical_cols:
            choice = st.selectbox(col, df[col].dropna().unique().tolist(), key=f"inp_{col}")
            inputs[col] = choice
        else:
            val = st.number_input(col, step=1.0, key=f"inp_{col}")
            inputs[col] = val
    submit = st.form_submit_button("Allocate Task")

# -------------------------
# Helper: convert user input to DataFrame
# -------------------------
def build_input_row(inputs):
    return pd.DataFrame([inputs], columns=feature_cols)

# -------------------------
# Allocation logic
# -------------------------
if submit:
    inp_df = build_input_row(inputs)
    prob_array = model.predict_proba(inp_df)[0]
    classes = model.classes_
    machine_probs = {str(c): float(prob_array[i]) for i, c in enumerate(classes)}

    def find_col_like(names):
        for n in feature_cols:
            for cand in names:
                if cand.lower() in n.lower():
                    return n
        return None

    est_col = find_col_like(["estimated", "estimated_time"])
    deadline_col = find_col_like(["deadline", "due"])
    priority_col = find_col_like(["priority"])

    est_time = float(inputs[est_col]) if est_col and est_col in inputs else 1.0
    deadline = float(inputs[deadline_col]) if deadline_col and deadline_col in inputs else est_time + 1.0
    priority_val = str(inputs[priority_col]) if priority_col and priority_col in inputs else "Medium"

    priority_map = {"high": 3.0, "medium": 1.0, "low": 0.0}
    priority_bonus = priority_map.get(priority_val.lower(), 1.0)

    prob_weight = 10.0
    slack_weight = 0.5
    load_weight = 1.0
    priority_factor = 2.0

    slack = max(0.0, deadline - est_time)

    candidate_scores = {}
    for m in machine_ids:
        m_key = str(m)
        current_load = float(st.session_state.machine_loads.get(m_key, 0))
        prob = machine_probs.get(m_key, 0.0)
        score = (
            load_weight * (current_load + est_time)
            + slack_weight * slack
            - prob_weight * prob
            - priority_factor * (priority_bonus)
        )
        candidate_scores[m_key] = score

    manpower_col = find_col_like(["manpower", "manpower_required"])
    manpower_req = int(inputs[manpower_col]) if manpower_col and manpower_col in inputs else 1

    available = int(st.session_state.manpower_available)
    if manpower_req > available:
        st.error(f"Not enough manpower: required {manpower_req}, available {available}")
    else:
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])
        assigned = sorted_candidates[0][0] if sorted_candidates else None

        if not assigned:
            st.error("No machine could be assigned.")
        else:
            st.session_state.machine_loads[assigned] += int(est_time)
            st.session_state.manpower_available -= int(manpower_req)
            task_summary = f"{inputs.get(find_col_like(['task','task_type']), '')} | load {inputs.get(find_col_like(['load','load_units']), '')} | est {est_time}h | priority {priority_val}"
            st.session_state.assigned_tasks[assigned].append(task_summary)

            st.success(f"✅ Allocated to Machine {assigned}")
            st.info(f"Machine {assigned} load now {st.session_state.machine_loads[assigned]} hrs. Manpower left: {st.session_state.manpower_available}")

if st.button("🔄 Reset System"):
    st.session_state.machine_loads = {str(m): 0 for m in machine_ids}
    st.session_state.manpower_available = 100
    st.session_state.assigned_tasks = {str(m): [] for m in machine_ids}
    st.success("System has been reset! All loads cleared, manpower restored.")

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

for mkey in sorted(st.session_state.assigned_tasks.keys(), key=lambda x: str(x)):
    with st.expander(f"Machine {mkey} tasks ({len(st.session_state.assigned_tasks[mkey])})"):
        for t in st.session_state.assigned_tasks[mkey]:
            st.write("-", t)
