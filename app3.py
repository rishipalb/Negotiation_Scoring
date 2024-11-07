import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up the app title
st.title("Efficient Frontier for Boeing and IAM with Custom Curvatures and Impact Visualization")

# Define default issues for both Boeing and IAM
default_boeing_issues = ["Wage Increase Flexibility", "Pension Reinstatement", "Healthcare Benefits", "Safety Protocol Control", "Job Security"]
default_iam_issues = ["Significant Wage Increase", "Pension Restoration", "Healthcare Improvements", "Overtime Protections", "Control over Safety Training"]

# Set parameters for S-shaped (sigmoid) and Default (exponential) curvatures
st.header("Negotiation: Rishipal Bansode, Michael Smith, Sundar Sundar")
st.header("Adjust Curvature Parameters")

# Sigmoid curvature parameters
st.subheader("Sigmoid (S-shaped) Curvature Parameters")
sigmoid_midpoint = st.slider("Sigmoid Midpoint (S-curve center)", 1.0, 10.0, 5.0)
sigmoid_steepness = st.slider("Sigmoid Steepness (S-curve slope)", 0.1, 2.0, 1.0)

# Default (exponential) curvature parameters
st.subheader("Default (Exponential) Curvature Parameters")
default_exponent = st.slider("Default Exponent (curve factor)", 0.5, 2.0, 1.0, step=0.1)

# Transformation functions for scores
def sigmoid(x, midpoint, steepness):
    return 1 / (1 + np.exp(-steepness * (x - midpoint)))

def exponential(x, exponent):
    return x ** exponent

def inverse_scaling(x, base_value, weight):
    """Inverse scaling for Boeing's wage-related score: higher score = lower impact."""
    return (1 / x) * base_value * weight

# Display sample curvature plots for Default (Exponential) and Sigmoid transformations
st.header("Sample Curvature Plots for Default and Sigmoid Transformations")

# Generate a range of sample scores for visualization
sample_scores = np.linspace(1, 10, 100)

# Plot the Default (Exponential) Curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
exponential_transformed = exponential(sample_scores, default_exponent)
plt.plot(sample_scores, exponential_transformed, label=f"Default (Exponent={default_exponent})", color="blue")
plt.xlabel("Score")
plt.ylabel("Transformed Score")
plt.title("Default (Exponential) Transformation")
plt.legend()

# Plot the Sigmoid (S-shaped) Curve
plt.subplot(1, 2, 2)
sigmoid_transformed = sigmoid(sample_scores, sigmoid_midpoint, sigmoid_steepness)
plt.plot(sample_scores, sigmoid_transformed, label="Sigmoid (S-shaped)", color="green")
plt.xlabel("Score")
plt.ylabel("Transformed Score")
plt.title("Sigmoid (S-shaped) Transformation")
plt.legend()

# Display the plots in Streamlit
st.pyplot(plt)

# Collect scores, weights, and base values from the user for each issue
st.header("Score, Weight, and Base Value for Each Issue")

boeing_scores = {}
iam_scores = {}
boeing_weights = {}
iam_weights = {}
boeing_base_values = {}
iam_base_values = {}
boeing_curvatures = {}
iam_curvatures = {}

st.subheader("Boeing Issues")
for issue in default_boeing_issues:
    boeing_scores[issue] = st.slider(f"{issue} Score", 1, 10, 5)
    boeing_weights[issue] = st.number_input(f"{issue} Weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    boeing_base_values[issue] = st.number_input(f"{issue} Base Value ($)", min_value=0.0, value=5000.0, step=100.0)
    boeing_curvatures[issue] = st.selectbox(f"{issue} Curvature Type", ["Default", "Sigmoid"])

st.subheader("IAM Issues")
for issue in default_iam_issues:
    iam_scores[issue] = st.slider(f"{issue} Score", 1, 10, 5)
    iam_weights[issue] = st.number_input(f"{issue} Weight", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    iam_base_values[issue] = st.number_input(f"{issue} Base Value ($)", min_value=0.0, value=5000.0, step=100.0)
    iam_curvatures[issue] = st.selectbox(f"{issue} Curvature Type", ["Default", "Sigmoid"])

# Define a small constant to avoid excessive reduction
small_constant = 1

# Define a small constant to adjust Boeing values and balance the impact
small_constant = 1
boeing_multiplier = 5  # To bring Boeing’s values closer to IAM's for comparable impact

# Calculate weighted scores and tangible values for Boeing and IAM based on selected curvatures
boeing_weighted_values = {}
iam_weighted_values = {}

# Reverse scaling for all Boeing issues
for issue in default_boeing_issues:
    base_value = boeing_base_values[issue]
    weight = boeing_weights[issue]
    
    # Apply reverse scaling with adjusted curvature formula for Boeing
    if boeing_curvatures[issue] == "Sigmoid":
        adjusted_score = (base_value * weight) / (sigmoid(boeing_scores[issue], sigmoid_midpoint, sigmoid_steepness) + small_constant)
    else:
        adjusted_score = (base_value * weight) / (exponential(boeing_scores[issue], default_exponent) + small_constant)
    
    boeing_weighted_values[issue] = adjusted_score

# Direct scaling for IAM issues, adjusted to match Boeing's scaling better
for issue in default_iam_issues:
    base_value = iam_base_values[issue]
    weight = iam_weights[issue]
    
    # Apply selected curvature type for IAM issues with a smaller weight for balance
    if iam_curvatures[issue] == "Sigmoid":
        adjusted_score = sigmoid(iam_scores[issue], sigmoid_midpoint, sigmoid_steepness) * base_value * weight / 5  # Adjusted for balance
    else:
        adjusted_score = exponential(iam_scores[issue], default_exponent) * base_value * weight / 5  # Adjusted for balance
    
    iam_weighted_values[issue] = adjusted_score

# Multiply Boeing’s total value by a factor to bring it closer to IAM’s
boeing_total_value = sum(boeing_weighted_values.values()) * boeing_multiplier
iam_total_value = sum(iam_weighted_values.values())

# Display tangible outputs for each issue
st.header("Tangible Outputs for Boeing and IAM")

st.subheader("Boeing Issues in Monetary Terms")
for issue, value in boeing_weighted_values.items():
    st.write(f"{issue}: ${value:,.2f}")
st.write(f"**Total Monetary Impact for Boeing's Issues: ${boeing_total_value:,.2f}**")

st.subheader("IAM Issues in Monetary Terms")
for issue, value in iam_weighted_values.items():
    st.write(f"{issue}: ${value:,.2f}")
st.write(f"**Total Monetary Impact for IAM's Issues: ${iam_total_value:,.2f}**")


# Efficient Frontier Calculation with Boeing and IAM values as coordinates
boeing_weight_range = np.linspace(0, 1, 50)
iam_weight_range = 1 - boeing_weight_range

frontier_scores = []

for w_boeing, w_iam in zip(boeing_weight_range, iam_weight_range):
    boeing_value = w_boeing * boeing_total_value
    iam_value = w_iam * iam_total_value
    combined_value = boeing_value + iam_value
    frontier_scores.append((boeing_value, iam_value, combined_value))

# Convert results to DataFrame for plotting
frontier_df = pd.DataFrame(frontier_scores, columns=["Boeing Value", "IAM Value", "Combined Value"])

# Plot Efficient Frontier as scatter plot with color indicating combined impact
st.header("Efficient Frontier Scatter Plot")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(frontier_df["Boeing Value"], frontier_df["IAM Value"],
                      c=frontier_df["Combined Value"], cmap="viridis", s=100, alpha=0.7)
plt.colorbar(scatter, label="Combined Monetary Impact")
plt.xlabel("Boeing Value (Weighted Monetary Impact)")
plt.ylabel("IAM Value (Weighted Monetary Impact)")
plt.title("Efficient Frontier: Boeing vs IAM with Combined Impact")
plt.grid()
plt.scatter(frontier_df["Boeing Value"], frontier_df["IAM Value"], color='blue', alpha=0.7)  # Add individual dots
st.pyplot(plt)

