import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
import time
import psutil
from datetime import datetime, timedelta

# Custom CSS to style the dashboard
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
            color: #1f2937;
        }
        .metric-label {
            color: #374151;
            font-size: 14px;
            font-weight: 500;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
        }
        .recent-alerts {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.random_anomaly_detector import RandomAnomalyDetector, AnomalyLogger, SystemHealthMonitor

# Define functions first (before they're used)
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

def plot_sensor_data(df, is_anomaly, selected_feature):
    """Create a plot showing sensor data with anomalies highlighted"""
    fig = go.Figure()
    
    # Plot all data points
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df[selected_feature],
        mode='lines+markers',
        name='Sensor Data',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))
    
    # Highlight anomalies
    if any(is_anomaly):
        anomaly_indices = np.where(is_anomaly)[0]
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=df.loc[anomaly_indices, selected_feature],
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=f'{selected_feature} Values with Anomaly Detection',
        xaxis_title='Data Point Index',
        yaxis_title=selected_feature,
        height=400
    )
    return fig

st.set_page_config(page_title="Pump Anomaly Detection", layout="wide")

# Initialize session state
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = RandomAnomalyDetector()
    
if 'anomaly_logger' not in st.session_state:
    st.session_state.anomaly_logger = AnomalyLogger()
    
if 'health_monitor' not in st.session_state:
    st.session_state.health_monitor = SystemHealthMonitor()
    
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
    
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
    
# Initialize empty data state
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = False

# Header section with system status
st.markdown("""
    <div class="main-header">
        <h1>🚨 Anomaly Detection Dashboard</h1>
        <p>Real-time monitoring and alert management for industrial systems</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**System Status:**")
with col2:
    system_status = st.session_state.get('streaming', False)
    has_data = st.session_state.get('uploaded_df') is not None and st.session_state.get('analysis_results', False)
    
    if system_status:
        st.button("🛑 Stop System", type="secondary", key="stop_system", 
                 on_click=lambda: setattr(st.session_state, 'streaming', False))
    else:
        if has_data:
            st.button("🟢 Start System", type="primary", key="start_system", 
                     on_click=lambda: setattr(st.session_state, 'streaming', True))
        else:
            st.button("⚠️ Upload Data First", type="secondary", key="start_system_disabled", 
                     disabled=True, help="Upload and analyze CSV data before starting streaming")

# Metrics section
col1, col2, col3, col4 = st.columns(4)

# Calculate dynamic metrics based on actual data
if st.session_state.get('analysis_results', False) and st.session_state.get('uploaded_df') is not None:
    # Use actual analysis results
    df = st.session_state.uploaded_df
    threshold = st.session_state.get('detection_threshold', 1.5)
    
    # Perform anomaly detection if not already done
    if 'anomaly_scores' not in st.session_state or 'is_anomaly' not in st.session_state:
        anomaly_scores, is_anomaly = detect_anomalies(df, threshold)
        st.session_state.anomaly_scores = anomaly_scores
        st.session_state.is_anomaly = is_anomaly
    
    # Get actual anomaly data
    total_alerts = sum(st.session_state.is_anomaly) if st.session_state.is_anomaly is not None else 0
    
    # Get user interactions
    confirmed_anomalies = len(st.session_state.get('confirmed_anomalies', set()))
    false_positives = len(st.session_state.get('false_positives', set()))
    pending_review = total_alerts - confirmed_anomalies - false_positives
    
    # Calculate accuracy based on confirmed vs total
    accuracy = (confirmed_anomalies / max(total_alerts, 1)) * 100 if total_alerts > 0 else 0.0
else:
    # No data uploaded yet - show zeros
    total_alerts = 0
    pending_review = 0
    confirmed_anomalies = 0
    false_positives = 0
    accuracy = 0.0

# Calculate response time (simulated based on system health)
avg_response_time = st.session_state.health_monitor.get_system_status().get('avg_processing_time', 0) if hasattr(st.session_state.health_monitor, 'get_system_status') else 0.0

# Calculate system uptime
uptime_seconds = (datetime.now() - st.session_state.start_time).total_seconds()
uptime_percentage = min(99.9, max(0, 100 - (uptime_seconds / (30 * 24 * 3600)) * 100))

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Alerts</div>
            <div class="metric-value">{total_alerts}</div>
            <div class="metric-label">{pending_review} pending review</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{accuracy:.1f}%</div>
            <div class="metric-label">{false_positives} false positives</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Response Time</div>
            <div class="metric-value">{avg_response_time:.1f}s</div>
            <div class="metric-label">Average response time</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">System Uptime</div>
            <div class="metric-value">{uptime_percentage:.1f}%</div>
            <div class="metric-label">Since startup</div>
        </div>
    """, unsafe_allow_html=True)

# Main content tabs
tabs = st.tabs(["Real-time Monitoring", "Alert Management", "Analytics", "System Status"])

# Main app content within tabs
with tabs[0]:  # Real-time Monitoring tab
    # File Upload Section
    st.subheader("📁 Upload Data for Analysis")
    
    # Show status message
    if st.session_state.get('uploaded_df') is None:
        st.info("📊 **No data uploaded yet** - Upload a CSV file to start anomaly detection analysis.")
    else:
        st.success(f"✅ **Data loaded** - {len(st.session_state.uploaded_df)} rows ready for analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file to analyze",
        type=['csv'],
        help="Upload a CSV file with sensor data for anomaly detection"
    )
    
    if uploaded_file is not None:
        try:
            # Load and preprocess the uploaded data
            df = load_and_preprocess_data(uploaded_file)
            st.success(f"✅ File uploaded successfully! Loaded {len(df)} rows and {len(df.columns)} columns.")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Anomaly Detection Controls
            st.subheader("🔍 Anomaly Detection Settings")
            col1, col2 = st.columns(2)
            with col1:
                detection_threshold = st.slider(
                    "Anomaly Threshold (IQR multiplier)",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Higher values = fewer anomalies detected"
                )
            with col2:
                col2a, col2b = st.columns(2)
                with col2a:
                    if st.button("🚀 Run Anomaly Detection", type="primary"):
                        st.session_state.analysis_results = True
                        st.session_state.uploaded_df = df
                        st.session_state.detection_threshold = detection_threshold
                        # Clear previous results
                        if 'anomaly_scores' in st.session_state:
                            del st.session_state.anomaly_scores
                        if 'is_anomaly' in st.session_state:
                            del st.session_state.is_anomaly
                        if 'confirmed_anomalies' in st.session_state:
                            del st.session_state.confirmed_anomalies
                        if 'false_positives' in st.session_state:
                            del st.session_state.false_positives
                        st.rerun()
                
                with col2b:
                    if st.button("🔄 Reset Analysis", type="secondary"):
                        st.session_state.analysis_results = False
                        st.session_state.uploaded_df = None
                        # Clear all analysis data
                        for key in ['anomaly_scores', 'is_anomaly', 'confirmed_anomalies', 'false_positives']:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
            
            # Run analysis if requested
            if st.session_state.get('analysis_results', False) and st.session_state.get('uploaded_df') is not None:
                df = st.session_state.uploaded_df
                threshold = st.session_state.detection_threshold
                
                # Perform anomaly detection
                anomaly_scores, is_anomaly = detect_anomalies(df, threshold)
                
                if anomaly_scores is not None:
                    # Display results
                    st.subheader("🎯 Anomaly Detection Results")
                    total_anomalies = sum(is_anomaly)
                    st.metric(
                        "Total Anomalies Detected",
                        total_anomalies,
                        f"{(total_anomalies/len(df)*100):.2f}% of data points"
                    )
                    
                    # Feature selection for visualization
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_cols) > 0:
                        selected_feature = st.selectbox(
                            "Select sensor to visualize",
                            options=numeric_cols,
                            help="Choose which sensor data to plot with anomaly highlights"
                        )
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Plot all data points
                        fig.add_trace(go.Scatter(
                            x=list(range(len(df))),
                            y=df[selected_feature],
                            mode='lines+markers',
                            name='Sensor Data',
                            line=dict(color='blue'),
                            marker=dict(size=4)
                        ))
                        
                        # Highlight anomalies
                        if total_anomalies > 0:
                            anomaly_indices = np.where(is_anomaly)[0]
                            fig.add_trace(go.Scatter(
                                x=anomaly_indices,
                                y=df.loc[anomaly_indices, selected_feature],
                                mode='markers',
                                name='Anomalies',
                                marker=dict(color='red', size=8, symbol='x'),
                                text=[f'Anomaly Score: {anomaly_scores[i]:.3f}' for i in anomaly_indices],
                                hovertemplate='<b>Anomaly</b><br>Index: %{x}<br>Value: %{y}<br>Score: %{text}<extra></extra>'
                            ))
                        
                        fig.update_layout(
                            title=f'{selected_feature} Values with Anomaly Detection',
                            xaxis_title='Data Point Index',
                            yaxis_title=selected_feature,
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Anomaly Review Section
                        st.subheader("🔍 Review Detected Anomalies")
                        if total_anomalies > 0:
                            st.info(f"Found {total_anomalies} anomalies. Review and confirm them below:")
                            
                            # Show anomaly details
                            anomaly_df = df[is_anomaly].copy()
                            anomaly_df['anomaly_score'] = anomaly_scores[is_anomaly]
                            anomaly_df['anomaly_index'] = np.where(is_anomaly)[0]
                            
                            # Display anomalies in an expandable section
                            with st.expander(f"📊 View {total_anomalies} Anomalies", expanded=True):
                                for idx, (_, row) in enumerate(anomaly_df.iterrows()):
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.write(f"**Anomaly {idx+1}** - Index: {row['anomaly_index']}, Score: {row['anomaly_score']:.3f}")
                                        st.write(f"Values: {', '.join([f'{col}: {row[col]:.2f}' for col in numeric_cols if col in row])}")
                                    
                                    with col2:
                                        if st.button(f"✅ Confirm", key=f"confirm_{idx}"):
                                            # Mark as confirmed
                                            if 'confirmed_anomalies' not in st.session_state:
                                                st.session_state.confirmed_anomalies = set()
                                            st.session_state.confirmed_anomalies.add(row['anomaly_index'])
                                            st.success("Anomaly confirmed!")
                                            st.rerun()
                                    
                                    with col3:
                                        if st.button(f"❌ False Positive", key=f"false_{idx}"):
                                            # Mark as false positive
                                            if 'false_positives' not in st.session_state:
                                                st.session_state.false_positives = set()
                                            st.session_state.false_positives.add(row['anomaly_index'])
                                            st.success("Marked as false positive!")
                                            st.rerun()
                                    
                                    st.divider()
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        results_df = df.copy()
                        results_df['anomaly_score'] = anomaly_scores
                        results_df['is_anomaly'] = is_anomaly
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Anomaly Detection Results (CSV)",
                            data=csv,
                            file_name="anomaly_detection_results.csv",
                            mime="text/csv",
                        )
                else:
                    st.error("❌ Anomaly detection failed. Please check your data format.")
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Make sure your CSV file has numeric columns for analysis.")
    
    # Two columns: Chart and Alerts
    chart_col, alerts_col = st.columns([2, 1])
    
    with chart_col:
        # Empty chart area - will be populated when data is uploaded
        st.info("📊 **Chart Area** - Upload a CSV file above to see anomaly detection visualizations here.")
        
        # Placeholder for future charts
        placeholder_chart = st.empty()
    
    with alerts_col:
        st.markdown("""
            <div class="recent-alerts">
                <h3>Recent Alerts</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Empty alerts area - will be populated when anomalies are detected
        st.info("🔔 **Alerts Area** - Anomalies detected in your data will appear here.")

# Sidebar controls for filtering and settings
with st.sidebar:
    st.title("Controls")
    
    st.subheader("Time Range")
    time_range = st.selectbox(
        "Select time range",
        ["Last 30 minutes", "Last 1 hour", "Last 24 hours", "Last 7 days"]
    )
    
    st.subheader("Alert Settings")
    threshold = st.slider("Anomaly threshold", 0.0, 1.0, 0.7)
    
    st.subheader("Data Source")
    data_source = st.selectbox(
        "Select data source",
        ["Sensor A", "Sensor B", "All Sensors"]
    )

# Initialize streaming data simulation
if st.session_state.streaming:
    st.session_state.current_time = datetime.now()
    if 'anomaly_scores' not in st.session_state:
        st.session_state.anomaly_scores = []
    if 'anomaly_times' not in st.session_state:
        st.session_state.anomaly_times = []

# Streaming simulation controls
streaming_section = st.sidebar.expander("Streaming Controls", expanded=True)
with streaming_section:
    if st.button("Toggle Streaming"):
        st.session_state.streaming = not st.session_state.streaming
    
    streaming_interval = st.slider(
        "Streaming Interval (seconds)",
        min_value=1,
        max_value=10,
        value=2
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Data")
    live_chart = st.empty()
    
    # Show placeholder message when no streaming data
    if not st.session_state.get('streaming', False):
        st.info("📡 **Live Sensor Data** - Enable streaming in the sidebar to see real-time sensor data here.")
    else:
        # Initialize empty plot for streaming data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Sensor Data'))
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Anomalies', marker=dict(color='red')))
        live_chart.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("System Health")
    health_metrics = st.empty()
    
    # Show placeholder message when no health data
    if not st.session_state.get('streaming', False):
        st.info("💻 **System Health** - Enable streaming to monitor CPU and memory usage.")

# Main content for other tabs
with tabs[1]:  # Alert Management
    st.header("Alert Management")
    st.write("Configure and manage alert settings here.")

with tabs[2]:  # Analytics
    st.header("Analytics")
    st.write("View detailed analytics and reports here.")

with tabs[3]:  # System Status
    st.header("System Status")
    st.write("Monitor system health and performance metrics here.")

# Streaming simulation - only works when real data has been uploaded
if st.session_state.streaming:
    # Check if we have real data uploaded and analyzed
    if st.session_state.get('uploaded_df') is not None and st.session_state.get('analysis_results', False):
        # Train the detector with uploaded data if not already done
        if st.session_state.anomaly_detector.training_data is None:
            st.session_state.anomaly_detector.train(st.session_state.uploaded_df)
        
        # Generate synthetic data point based on real data
        synthetic_data = st.session_state.anomaly_detector.generate_synthetic_data(1)
        
        # Initialize data buffer if not exists
        if 'data_buffer' not in st.session_state:
            st.session_state.data_buffer = synthetic_data
        else:
            st.session_state.data_buffer = pd.concat([st.session_state.data_buffer, synthetic_data])
            # Keep only last 100 points
            if len(st.session_state.data_buffer) > 100:
                st.session_state.data_buffer = st.session_state.data_buffer.iloc[-100:]
        
        # Detect anomalies
        anomalies, confidence = st.session_state.anomaly_detector.detect_anomalies(synthetic_data)
        
        # Update live chart
        with col1:
            fig = go.Figure()
        
        # Get the first numeric column for visualization
        numeric_col = st.session_state.anomaly_detector.numeric_columns[0]
        
        # Create timestamps for x-axis if not present
        if st.session_state.anomaly_detector.datetime_columns.empty:
            # Always update the current timestamp
            st.session_state.timestamp = pd.Timestamp.now()
            
            # Generate a sequence of timestamps ending at current time
            timestamps = pd.date_range(
                end=st.session_state.timestamp,
                periods=len(st.session_state.data_buffer),
                freq='S'
            )
            x_values = timestamps
        else:
            time_col = st.session_state.anomaly_detector.datetime_columns[0]
            x_values = st.session_state.data_buffer[time_col]
        
        # Ensure x_values is a pandas Series or numpy array for consistent plotting
        if hasattr(x_values, 'iloc'):
            x_values = x_values.values  # Convert pandas Series to numpy array
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=x_values,
            y=st.session_state.data_buffer[numeric_col].values,  # Ensure y is also numpy array
            mode='lines',
            name='Sensor Data'
        ))
        
        if anomalies[0]:
            # Plot anomaly point - ensure consistent data types
            anomaly_x = x_values[-1] if len(x_values) > 0 else x_values[0]
            anomaly_y = synthetic_data[numeric_col].iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[anomaly_x],
                y=[anomaly_y],
                mode='markers',
                name='Anomaly',
                marker={'color': 'red', 'size': 10}
            ))
            
            # Log anomaly
            st.session_state.anomaly_logger.log_anomaly(
                timestamp=datetime.now(),
                features=synthetic_data.iloc[-1].to_dict(),
                confidence=confidence
            )
            
            # Show anomaly alert
            st.warning(f"⚠️ Anomaly detected! Confidence: {confidence:.2f}")
            
            if st.button("Confirm Anomaly"):
                st.session_state.anomaly_logger.log_anomaly(
                    timestamp=datetime.now(),
                    features=synthetic_data.iloc[-1].to_dict(),
                    is_confirmed=True,
                    confidence=confidence
                )
                st.success("Anomaly confirmed and logged!")
        
        # Update chart layout
        fig.update_layout(
            title=f'{numeric_col} Values Over Time',
            xaxis_title='Time',
            yaxis_title=numeric_col
        )
        live_chart.plotly_chart(fig, use_container_width=True)
    
        # Update system health metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        st.session_state.health_monitor.record_metric(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            processing_time=time.time() % 60  # Example processing time
        )
        
        health_status = st.session_state.health_monitor.get_system_status()
        with col2:
            health_metrics.metric(
                "CPU Usage",
                f"{cpu_usage:.1f}%",
                f"{health_status.get('avg_cpu', 0):.1f}% avg"
            )
            health_metrics.metric(
                "Memory Usage",
                f"{memory_usage:.1f}%",
                f"{health_status.get('avg_memory', 0):.1f}% avg"
            )
        
        time.sleep(streaming_interval)
        st.rerun()
    else:
        # Show message that real data is needed for streaming
        st.warning("⚠️ **Streaming requires real data** - Please upload a CSV file and run anomaly detection first.")
        st.session_state.streaming = False  # Auto-stop streaming
        st.rerun()
