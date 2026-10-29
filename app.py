
import numpy as np
import streamlit as st

# -------------------------------
# 1. Train a model using NumPy
# -------------------------------

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)

    # Synthetic training data: hours studied and attendance
    hours = rng.uniform(0, 10, 1000)
    attendance = rng.uniform(50, 100, 1000)

    X = np.column_stack((hours, attendance))

    # Synthetic scores based on an assumed relationship
    noise = rng.normal(0, 5, 1000)
    y = np.clip(
        25 + 5 * hours + 0.45 * (attendance - 50) + noise,
        0,
        100,
    )

    # Standardize the input features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_scaled = (X - mean) / std

    # Add a bias/intercept column
    X_bias = np.column_stack(
        (np.ones(len(X_scaled)), X_scaled)
    )

    # Initialize weights
    weights = np.zeros(X_bias.shape[1])
    learning_rate = 0.05
    epochs = 3000
    n = len(y)

    # Gradient descent
    for _ in range(epochs):
        predictions = X_bias @ weights
        errors = predictions - y
        gradient = (2 / n) * (X_bias.T @ errors)
        weights -= learning_rate * gradient

    # Training-set error for demonstration
    final_predictions = X_bias @ weights
    mse = np.mean((final_predictions - y) ** 2)

    return weights, mean, std, mse


def predict_score(hours, attendance, model):
    weights, mean, std, _ = model

    X = np.array([[hours, attendance]], dtype=float)
    X_scaled = (X - mean) / std
    X_bias = np.column_stack((np.ones(1), X_scaled))

    score = float((X_bias @ weights)[0])
    return float(np.clip(score, 0, 100))


# -------------------------------
# 2. Streamlit user interface
# -------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Student Performance Predictor")
st.write(
    "Explore how study hours and attendance affect "
    "a score predicted by a simple NumPy model."
)

model = train_model()

st.sidebar.header("Student inputs")

hours = st.sidebar.slider(
    "Study hours per day",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=50,
    max_value=100,
    value=80,
    step=1,
)

score = predict_score(hours, attendance, model)

st.subheader("Prediction")

col1, col2 = st.columns(2)

col1.metric("Predicted score", f"{score:.1f}/100")
col2.metric("Study hours/day", f"{hours:.1f}")

st.progress(int(round(score)))

if score >= 75:
    st.success("The toy model predicts a higher score.")
elif score >= 50:
    st.info("The toy model predicts a middle-range score.")
else:
    st.warning("The toy model predicts a lower score.")

st.subheader("Model information")

st.write("Algorithm: Linear Regression")
st.write("Optimization: Gradient Descent")
st.write("ML library: NumPy")
st.write(f"Training-set MSE: {model[3]:.2f}")

st.caption(
    "Educational demonstration only. Scores are generated "
    "from synthetic data and are not reliable predictions "
    "of any real student's academic performance."
)
