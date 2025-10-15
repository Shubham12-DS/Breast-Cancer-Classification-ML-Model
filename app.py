import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🎗️",
    layout="wide",
)

# --- LOAD THE MODEL ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the pickled logistic regression model."""
    try:
        with open('best_LR.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'best_LR.pkl' not found. Please make sure it's in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()


# --- SIDEBAR FOR INPUTS ---
st.sidebar.header("🔬 Input Tumor Features")
st.sidebar.markdown("Use the sliders below to input the feature values for prediction.")

def user_input_features():
    """Creates sliders in the sidebar for user input."""
    # Define features and their typical ranges (based on the standard dataset)
    # This helps in setting logical min, max, and default values for sliders
    feature_ranges = {
        'radius_mean': (5.0, 30.0, 14.1), 'texture_mean': (9.0, 40.0, 19.3),
        'perimeter_mean': (40.0, 190.0, 92.0), 'area_mean': (140.0, 2500.0, 655.0),
        'smoothness_mean': (0.05, 0.16, 0.096), 'compactness_mean': (0.02, 0.35, 0.104),
        'concavity_mean': (0.0, 0.43, 0.088), 'concave points_mean': (0.0, 0.2, 0.048),
        'symmetry_mean': (0.1, 0.3, 0.181), 'fractal_dimension_mean': (0.05, 0.1, 0.062),
        'radius_se': (0.1, 2.9, 0.405), 'texture_se': (0.3, 4.9, 1.21),
        'perimeter_se': (0.7, 22.0, 2.86), 'area_se': (6.0, 542.0, 40.3),
        'smoothness_se': (0.001, 0.031, 0.007), 'compactness_se': (0.002, 0.135, 0.025),
        'concavity_se': (0.0, 0.4, 0.031), 'concave points_se': (0.0, 0.052, 0.011),
        'symmetry_se': (0.008, 0.079, 0.02), 'fractal_dimension_se': (0.001, 0.03, 0.003),
        'radius_worst': (7.0, 36.0, 16.2), 'texture_worst': (12.0, 50.0, 25.6),
        'perimeter_worst': (50.0, 251.0, 107.2), 'area_worst': (180.0, 4250.0, 880.0),
        'smoothness_worst': (0.07, 0.22, 0.132), 'compactness_worst': (0.02, 1.06, 0.254),
        'concavity_worst': (0.0, 1.25, 0.272), 'concave points_worst': (0.0, 0.29, 0.114),
        'symmetry_worst': (0.15, 0.66, 0.29), 'fractal_dimension_worst': (0.05, 0.21, 0.083)
    }

    data = {}
    for feature, (min_val, max_val, default_val) in feature_ranges.items():
        # Create a more user-friendly label
        label = feature.replace('_', ' ').title()
        data[feature] = st.sidebar.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step= (max_val - min_val) / 1000 # a reasonable step
        )

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- MAIN PANEL ---
st.title("🎗️ Breast Cancer Classification")
st.markdown(
    "This application uses a Logistic Regression model to predict whether a breast tumor is **benign** (non-cancerous) or **malignant** (cancerous). "
    "The prediction updates in real-time as you adjust the feature values in the sidebar."
)
st.markdown("---")


# Display User Inputs
st.subheader("Current Input Features")
st.write("The table below shows the current values from the sidebar sliders.")
st.dataframe(input_df, hide_index=True)


# Prediction Output
st.subheader("Prediction Result")

if model is not None:
    # Ensure the column order is the same as the model was trained on
    try:
        feature_names = model.feature_names_in_
        
        # Create a copy for prediction to avoid altering the displayed dataframe
        prediction_df = input_df.copy()

        # WORKAROUND: The saved model expects an 'id' column that was likely in the
        # training data but is not a predictive feature. We add a placeholder
        # column to match the model's expected input shape.
        if 'id' in feature_names and 'id' not in prediction_df.columns:
            prediction_df['id'] = 0 # Add a dummy 'id' column with a placeholder value

        input_df_ordered = prediction_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df_ordered)
        prediction_proba = model.predict_proba(input_df_ordered)

        # Display result using Streamlit's native components for a cleaner look
        if prediction[0] == 'B':
            st.success(f"**Prediction: Benign (Non-Cancerous)**")
            st.metric(label="Confidence", value=f"{prediction_proba[0][0]*100:.2f}%")
            st.info(
                "**Disclaimer:** This is a prediction from a machine learning model and should not be considered a medical diagnosis. "
                "Always consult a qualified healthcare professional for medical advice."
            )
        else:
            st.error(f"**Prediction: Malignant (Cancerous)**")
            st.metric(label="Confidence", value=f"{prediction_proba[0][1]*100:.2f}%")
            st.warning(
                "**Disclaimer:** This is a prediction from a machine learning model and should not be considered a medical diagnosis. "
                "Always consult a qualified healthcare professional for medical advice."
            )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Model could not be loaded. Prediction cannot be performed.")

st.markdown("---")
st.write("Created with Streamlit and Scikit-learn.")

