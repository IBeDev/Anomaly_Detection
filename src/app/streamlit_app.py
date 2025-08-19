import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Pump Anomaly Detection", layout="wide")

st.title("Pump Sensor Anomaly Detection")
st.write("""
This application helps detect anomalies in pump sensor data. Upload your CSV file containing sensor readings 
to visualize and detect anomalies in the pump's behavior.
""")

def load_and_preprocess_data(file):
    """Load and preprocess the uploaded data"""
    df = pd.read_csv(file)
    
    # Convert datetime column if it exists
    datetime_cols = df.select_dtypes(include=['object']).columns
    for col in datetime_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except (ValueError, TypeError):
            continue
    
    return df

def detect_anomalies(df, threshold=1.5):
    """
    Detect anomalies using the IQR method
    Returns DataFrame with anomaly scores and boolean mask of anomalies
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_df = df[numeric_cols]
    
    if len(numeric_cols) == 0:
        st.error("No numeric columns found in the data. Please check your input file.")
        return None, None
    
    # Calculate z-scores for each feature
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    z_scores = pd.DataFrame(scaled_data, columns=numeric_cols)
    
    # Calculate anomaly scores (mean absolute z-score across features)
    anomaly_scores = np.mean(np.abs(z_scores), axis=1)
    
    # Define anomalies based on IQR method
    Q1 = np.percentile(anomaly_scores, 25)
    Q3 = np.percentile(anomaly_scores, 75)
    IQR = Q3 - Q1
    anomaly_threshold = Q3 + threshold * IQR
    
    is_anomaly = anomaly_scores > anomaly_threshold
    
    return anomaly_scores, is_anomaly

def plot_sensor_data(df, anomaly_mask, feature):
    """Create an interactive plot of sensor data with anomalies highlighted"""
    fig = px.scatter(
        x=range(len(df)),
        y=df[feature],
        color=anomaly_mask,
        labels={'x': 'Time Point', 'y': f'{feature} Value'},
        title=f'{feature} Values Over Time',
        color_discrete_map={True: 'red', False: 'blue'}
    )
    return fig

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file with sensor data", type="csv")

if uploaded_file is not None:
    # Load and process data
    df = load_and_preprocess_data(uploaded_file)
    
    # Show basic data information
    st.subheader("Data Overview")
    st.write(f"Number of readings: {len(df)}")
    st.write(f"Number of sensors: {len(df.columns)}")
    
    # Allow user to configure anomaly detection
    st.subheader("Anomaly Detection Configuration")
    threshold = st.slider(
        "IQR Threshold Multiplier",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Higher values mean fewer anomalies will be detected"
    )
    
    # Perform anomaly detection
    anomaly_scores, is_anomaly = detect_anomalies(df, threshold)
    
    # Display results
    st.subheader("Anomaly Detection Results")
    total_anomalies = sum(is_anomaly)
    st.write(f"Found {total_anomalies} anomalies ({(total_anomalies/len(df)*100):.2f}% of data points)")
    
    # Feature selection for visualization
    selected_feature = st.selectbox(
        "Select sensor to visualize",
        options=df.columns
    )
    
    # Plot the selected feature
    st.plotly_chart(plot_sensor_data(df, is_anomaly, selected_feature), use_container_width=True)
    
    # Download results
    if st.button("Download Results"):
        results_df = df.copy()
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = is_anomaly
        
        # Convert DataFrame to CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name="anomaly_detection_results.csv",
            mime="text/csv",
        )
