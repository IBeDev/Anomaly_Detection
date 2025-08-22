import pandas as pd
import numpy as np
from datetime import datetime

class RandomAnomalyDetector:
    def __init__(self):
        self.training_data = None
        self.numeric_columns = None
        self.datetime_columns = None
        self.means = None
        self.stds = None

    def train(self, df):
        """Train the detector with historical data"""
        self.training_data = df
        self.numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        self.datetime_columns = pd.Index([col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])])
        
        # Calculate statistics for numeric columns
        self.means = df[self.numeric_columns].mean()
        self.stds = df[self.numeric_columns].std()

    def generate_synthetic_data(self, n_points=1):
        """Generate synthetic data points based on training data distribution"""
        if self.training_data is None:
            raise ValueError("Detector must be trained first")
        
        synthetic_data = pd.DataFrame()
        
        # Generate data for numeric columns
        for col in self.numeric_columns:
            # Add random noise to mean
            synthetic_data[col] = np.random.normal(
                loc=self.means[col],
                scale=self.stds[col],
                size=n_points
            )
        
        # Add timestamp if datetime columns exist
        for col in self.datetime_columns:
            synthetic_data[col] = pd.date_range(
                end=pd.Timestamp.now(),
                periods=n_points
            )
        
        return synthetic_data

    def detect_anomalies(self, data, threshold=3.0):
        """
        Detect anomalies in new data points
        Returns tuple (is_anomaly, confidence)
        """
        if self.training_data is None:
            raise ValueError("Detector must be trained first")
        
        # Calculate z-scores for numeric columns
        z_scores = np.abs((data[self.numeric_columns] - self.means) / self.stds)
        
        # Calculate mean z-score across features
        mean_z_score = z_scores.mean(axis=1)
        
        # Mark as anomaly if mean z-score exceeds threshold
        is_anomaly = mean_z_score > threshold
        
        # Calculate confidence based on how far the z-score is above threshold
        confidence = np.clip((mean_z_score - threshold) / threshold, 0, 1)
        
        return is_anomaly.values, confidence.values[0] if len(confidence) > 0 else 0.0

class AnomalyLogger:
    def __init__(self):
        self.logs = []

    def log_anomaly(self, timestamp, features, confidence, is_confirmed=False):
        """Log an anomaly event"""
        self.logs.append({
            'timestamp': timestamp,
            'features': features,
            'confidence': confidence,
            'is_confirmed': is_confirmed
        })

    def get_recent_logs(self, n=10):
        """Get most recent anomaly logs"""
        return sorted(self.logs, key=lambda x: x['timestamp'], reverse=True)[:n]

class SystemHealthMonitor:
    def __init__(self):
        self.metrics = []
        self.window_size = 60  # Keep last 60 measurements

    def record_metric(self, timestamp, cpu_usage, memory_usage, processing_time):
        """Record system health metrics"""
        self.metrics.append({
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'processing_time': processing_time
        })
        
        # Keep only recent metrics
        if len(self.metrics) > self.window_size:
            self.metrics = self.metrics[-self.window_size:]

    def get_system_status(self):
        """Get average system metrics"""
        if not self.metrics:
            return {
                'avg_cpu': 0,
                'avg_memory': 0,
                'avg_processing_time': 0
            }
        
        recent_metrics = self.metrics[-self.window_size:]
        return {
            'avg_cpu': sum(m['cpu_usage'] for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m['memory_usage'] for m in recent_metrics) / len(recent_metrics),
            'avg_processing_time': sum(m['processing_time'] for m in recent_metrics) / len(recent_metrics)
        }
