import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set up the app title
st.title("Efficient Frontier for Boeing and IAM with Custom Curvatures and Impact Visualization")

# Define all Boeing and IAM issues
boeing_issues = ["Wage Increase Flexibility", "Pension Reinstatement", "Healthcare Benefits", 
                 "Safety Protocol Control", "Job Security", "Refresher Training", 
                 "Quality Initiatives", "Recertification", "Supply Chain Bottlenecks"]
iam_issues = ["Significant Wage Increase", "Pension Restoration", "Healthcare Improvements", 
              "Overtime Protections", "Control over Safety Training"]

# Display team members
st.header("Negotiation: Rishipal Bansode, Michael Smith, Sundar Sundar")

# Sigmoid and Exponential parameters
st.header("Adjust Curvature Parameters")

# Sigmoid curvature parameters
st.subheader("Sigmoid (S-shaped) Curvature Parameters")
sigmoid_midpoint = st.slider("Sigmoid Midpoint (S-curve center)", 1.0, 10.0, 5.0)
sigmoid_steepness = st.slider("Sigmoid Steepness (S-curve slope)", 0.1, 2.0, 1.0)

# Default (exponential) curvature parameters
st.subheader("Default (Exponential) Curvature Parameters")
default_exponent = st.slider("Default Exponent (curve factor)", 0.5, 2.0, 1.0, step=0.1)

# Transformation functions
def sigmoid(x, midpoint, steepness):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def exponential(x, exponent):
    return x ** exponent

# Display sample curvature plots
st.header("Sample Curvature Plots for Default and Sigmoid Transformations")
sample_scores = np.linspace(1, 10, 100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
exponential_transformed = exponential(sample_scores, default_exponent)
plt.plot(sample_scores, exponential_transformed, label=f"Default (Exponent={default_exponent})", color="blue")
plt.xlabel("Score")
plt.ylabel("Transformed Score")
plt.title("Default (Exponential) Transformation")
plt.legend()

plt.subplot(1, 2, 2)
sigmoid_transformed = sigmoid(sample_scores, sigmoid_midpoint, sigmoid_steepness)
plt.plot(sample_scores, sigmoid_transformed, label="Sigmoid (S-shaped)", color="green")
plt.xlabel("Score")
plt.ylabel("Transformed Score")
plt.title("Sigmoid (S-shaped) Transformation")
plt.legend()

st.pyplot(plt)

# Collect inputs
st.header("Score, Weight, Base Value, and Curvature Type for Each Issue")

boeing_scores = {}
iam_scores = {}
boeing_weights = {}
iam_weights = {}
boeing_base_values = {}
iam_base_values = {}
boeing_curvatures = {}
iam_curvatures = {}

# Boeing Issues - Input
st.subheader("Boeing Issues")
for issue in boeing_issues:
    boeing_scores[issue] = st.slider(f"{issue} Score", 1, 10, 5, 
                                     help="Higher score = more efficiency = lower costs")
    boeing_weights[issue] = st.number_input(f"{issue} Weight", min_value=0.0, max_value=0.15, value=0.10, step=0.01)
    boeing_base_values[issue] = st.number_input(f"{issue} Base Value ($)", min_value=0.0, value=8000.0, step=1000.0)
    boeing_curvatures[issue] = st.selectbox(f"{issue} Curvature Type", ["Default (Exponential)", "Sigmoid (S-shaped)"])

# IAM Issues - Input
st.subheader("IAM Issues")
for issue in iam_issues:
    iam_scores[issue] = st.slider(f"{issue} Score", 1, 10, 5)
    iam_weights[issue] = st.number_input(f"{issue} Weight", min_value=0.0, max_value=0.25, value=0.20, step=0.01)
    iam_base_values[issue] = st.number_input(f"{issue} Base Value ($)", min_value=0.0, value=12000.0, step=1000.0)
    iam_curvatures[issue] = st.selectbox(f"{issue} Curvature Type", ["Default (Exponential)", "Sigmoid (S-shaped)"])

# Calculate weighted values
boeing_weighted_values = {}
iam_weighted_values = {}

# Boeing calculations with reverse scaling
for issue in boeing_issues:
    score = boeing_scores[issue]
    weight = boeing_weights[issue]
    base_value = boeing_base_values[issue]
    
    if boeing_curvatures[issue] == "Sigmoid (S-shaped)":
        adjusted_score = sigmoid(11 - score, sigmoid_midpoint, sigmoid_steepness)
    else:
        # Normalize score before exponential transformation
        adjusted_score = exponential((11 - score) / 10.0, default_exponent)
    
    boeing_weighted_values[issue] = base_value * weight * adjusted_score

# IAM calculations
for issue in iam_issues:
    score = iam_scores[issue]
    weight = iam_weights[issue]
    base_value = iam_base_values[issue]
    
    if iam_curvatures[issue] == "Sigmoid (S-shaped)":
        adjusted_score = sigmoid(score, sigmoid_midpoint, sigmoid_steepness)
    else:
        # Normalize score before exponential transformation
        adjusted_score = exponential(score / 10.0, default_exponent)
    
    iam_weighted_values[issue] = base_value * weight * adjusted_score

# Calculate totals
boeing_total_value = sum(boeing_weighted_values.values())
iam_total_value = sum(iam_weighted_values.values())

# Display outputs
st.header("Tangible Outputs for Boeing and IAM")

st.subheader("Boeing Issues in Monetary Terms (Cost Savings)")
for issue, value in boeing_weighted_values.items():
    st.write(f"{issue}: ${value:,.2f}")
st.write(f"**Total Cost Savings Impact for Boeing's Issues: ${boeing_total_value:,.2f}**")

st.subheader("IAM Issues in Monetary Terms (Costs)")
for issue, value in iam_weighted_values.items():
    st.write(f"{issue}: ${value:,.2f}")
st.write(f"**Total Cost Impact for IAM's Issues: ${iam_total_value:,.2f}**")

# Efficient Frontier calculation and plotting
boeing_weight_range = np.linspace(0, 1, 50)
iam_weight_range = 1 - boeing_weight_range

frontier_scores = []

for w_boeing, w_iam in zip(boeing_weight_range, iam_weight_range):
    boeing_value = w_boeing * boeing_total_value
    iam_value = w_iam * iam_total_value
    combined_value = boeing_value + iam_value
    frontier_scores.append((boeing_value, iam_value, combined_value))

frontier_df = pd.DataFrame(frontier_scores, columns=["Boeing Value", "IAM Value", "Combined Value"])

st.header("Efficient Frontier Scatter Plot")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(frontier_df["Boeing Value"], frontier_df["IAM Value"],
                      c=frontier_df["Combined Value"], cmap="viridis", s=100, alpha=0.7)
plt.colorbar(scatter, label="Combined Monetary Impact")
plt.xlabel("Boeing Cost Savings (Weighted Monetary Impact)")
plt.ylabel("IAM Cost Impact (Weighted Monetary Impact)")
plt.title("Efficient Frontier: Boeing Cost Savings vs IAM Costs")
plt.grid()
plt.scatter(frontier_df["Boeing Value"], frontier_df["IAM Value"], color='blue', alpha=0.7)
st.pyplot(plt)
