import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

st.set_page_config(page_title="Adaptive Scheduling", layout="wide")
st.title("Adaptive Scheduling — AI + Adaptive Allocation (Random Forest)")

# -------------------------
# Upload dataset
# -------------------------
uploaded_file = st.file_uploader("Upload your scheduling dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV with your historical jobs (must include a machine column).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Dataset preview", df.head())

# -------------------------
# Let user choose target and features
# -------------------------
target_col = st.selectbox(
    "Select target column (machine ID)",
    df.columns,
    index=df.columns.get_loc("Machine_Available") if "Machine_Available" in df.columns else 0
)
ignore_cols = st.multiselect("Columns to ignore as features (identifiers)", ["Job_ID"], default=["Job_ID"])
feature_cols = [c for c in df.columns if c != target_col and c not in ignore_cols]
st.write("Using features:", feature_cols)

# -------------------------
# Encode categorical features
# -------------------------
label_encoders = {}
df_enc = df.copy()
for col in df_enc.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

X = df_enc[feature_cols]
y = df_enc[target_col]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train Random Forest model
# -------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
joblib.dump(model, "scheduling_model_rf.pkl")

# -------------------------
# Evaluate model
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success("Model trained (Random Forest).")
st.write(f"**Accuracy on test set:** {acc:.4f}")

# -------------------------
# Session state initialization
# -------------------------
machine_ids = sorted([int(x) for x in np.unique(df_enc[target_col])])

if "machine_loads" not in st.session_state:
    st.session_state.machine_loads = {str(m): 0 for m in machine_ids}
if "manpower_available" not in st.session_state:
    st.session_state.manpower_available = 100
if "assigned_tasks" not in st.session_state:
    st.session_state.assigned_tasks = {str(m): [] for m in machine_ids}

# -------------------------
# Real-time input (form)
# -------------------------
st.subheader("Real-time Task Scheduling")
with st.form("task_form"):
    inputs = {}
    for col in feature_cols:
        if col in label_encoders:
            choice = st.selectbox(col, label_encoders[col].classes_.tolist(), key=f"inp_{col}")
            inputs[col] = choice
        else:
            val = st.number_input(col, step=1.0, key=f"inp_{col}")
            inputs[col] = val
    submit = st.form_submit_button("Allocate Task")

# -------------------------
# Helper: transform user input
# -------------------------
def build_input_row(inputs):
    row = {}
    for col in feature_cols:
        if col in label_encoders:
            row[col] = int(label_encoders[col].transform([str(inputs[col])])[0])
        else:
            row[col] = float(inputs[col])
    return pd.DataFrame([row], columns=feature_cols)

# -------------------------
# Allocation logic
# -------------------------
if submit:
    try:
        inp_df = build_input_row(inputs)
    except Exception as e:
        st.error("Failed to encode inputs: " + str(e))
        st.stop()

    prob_array = model.predict_proba(inp_df)[0]
    classes = model.classes_.astype(int)
    machine_probs = {int(c): float(prob_array[i]) for i, c in enumerate(classes)}

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
        m_key = str(int(m))
        current_load = float(st.session_state.machine_loads.get(m_key, 0))
        prob = machine_probs.get(int(m), 0.0)
        score = (
            load_weight * (current_load + est_time)
            + slack_weight * slack
            - prob_weight * prob
            - priority_factor * (priority_bonus)
        )
        candidate_scores[int(m)] = score

    manpower_col = find_col_like(["manpower", "manpower_required"])
    manpower_req = int(inputs[manpower_col]) if manpower_col and manpower_col in inputs else 1

    available = int(st.session_state.manpower_available)
    if manpower_req > available:
        st.error(f"Not enough manpower: required {manpower_req}, available {available}")
    else:
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1])
        assigned = None
        for m, sc in sorted_candidates:
            assigned = m
            break

        if assigned is None:
            st.error("No machine could be assigned.")
        else:
            mkey = str(int(assigned))
            st.session_state.machine_loads[mkey] = int(st.session_state.machine_loads.get(mkey, 0)) + int(est_time)
            st.session_state.manpower_available = int(st.session_state.manpower_available) - int(manpower_req)
            task_summary = f"{inputs.get(find_col_like(['task','task_type']), '')} | load {inputs.get(find_col_like(['load','load_units']), '')} | est {est_time}h | priority {priority_val}"
            st.session_state.assigned_tasks[mkey].append(task_summary)

            st.success(f"✅ Allocated to Machine {assigned}")
            st.info(f"Machine {assigned} load now {st.session_state.machine_loads[mkey]} hrs. Manpower left: {st.session_state.manpower_available}")

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

for mkey in sorted(st.session_state.assigned_tasks.keys(), key=lambda x: int(x)):
    with st.expander(f"Machine {mkey} tasks ({len(st.session_state.assigned_tasks[mkey])})"):
        for t in st.session_state.assigned_tasks[mkey]:
            st.write("-", t)
