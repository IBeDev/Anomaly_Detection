import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List

class RandomAnomalyDetector:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.model_state = {}
        self.training_data = None
        self.numeric_columns = None
        self.datetime_columns = None
        
    def train(self, data: pd.DataFrame) -> None:
        """Placeholder for model training"""
        self.training_data = data
        
        # Identify numeric and datetime columns
        self.numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        self.datetime_columns = data.select_dtypes(include=['datetime64']).columns
        
        # Store basic statistics for synthetic data generation (numeric columns only)
        numeric_data = data[self.numeric_columns]
        self.model_state = {
            'mean': numeric_data.mean(),
            'std': numeric_data.std(),
            'min': numeric_data.min(),
            'max': numeric_data.max()
        }
        
        # Store datetime range if datetime columns exist
        if not self.datetime_columns.empty:
            self.model_state['datetime_range'] = {
                col: (data[col].min(), data[col].max())
                for col in self.datetime_columns
            }
    
    def detect_anomalies(self, data: pd.DataFrame) -> Tuple[np.ndarray, float]:
        """
        Detect anomalies using random probability
        Returns:
            - Boolean array of anomalies
            - Confidence score
        """
        # Random probability for each point
        rng = np.random.default_rng()
        probabilities = rng.random(len(data))
        anomalies = probabilities > self.threshold
        confidence = np.mean(probabilities[anomalies]) if any(anomalies) else 0
        
        return anomalies, confidence
    
    def generate_synthetic_data(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic data similar to training data"""
        if self.training_data is None:
            raise ValueError("Model must be trained before generating synthetic data")
        
        rng = np.random.default_rng()    
        synthetic_data = {}
        
        # Generate synthetic values for numeric columns
        for column in self.numeric_columns:
            mean = self.model_state['mean'][column]
            std = self.model_state['std'][column]
            min_val = self.model_state['min'][column]
            max_val = self.model_state['max'][column]
            
            # Generate data within the observed range
            synthetic_values = rng.normal(mean, std, n_samples)
            synthetic_values = np.clip(synthetic_values, min_val, max_val)
            synthetic_data[column] = synthetic_values
        
        # Generate synthetic timestamps for datetime columns
        for column in self.datetime_columns:
            min_time, max_time = self.model_state['datetime_range'][column]
            min_ts = min_time.timestamp()
            max_ts = max_time.timestamp()
            
            # Generate random timestamps within the observed range
            random_timestamps = rng.uniform(min_ts, max_ts, n_samples)
            synthetic_data[column] = [pd.Timestamp.fromtimestamp(ts) for ts in random_timestamps]
            
        return pd.DataFrame(synthetic_data)

class AnomalyLogger:
    def __init__(self, log_file: str = "anomaly_log.csv"):
        self.log_file = log_file
        self.log_data = []
        
    def log_anomaly(self, timestamp: datetime, features: Dict[str, float], 
                    is_confirmed: bool = None, confidence: float = None) -> None:
        """Log detected anomaly with optional confirmation"""
        log_entry = {
            'timestamp': timestamp,
            'features': features,
            'is_confirmed': is_confirmed,
            'confidence': confidence
        }
        self.log_data.append(log_entry)
        # Don't save during initialization to avoid file I/O errors
        if len(self.log_data) > 3:  # Only save after initial sample data
            self._save_log()
    
    def get_anomalies(self) -> List[Dict]:
        """Get all logged anomalies"""
        return self.log_data
    
    def _save_log(self) -> None:
        """Save log entries to CSV file"""
        df = pd.DataFrame(self.log_data)
        df.to_csv(self.log_file, index=False)
    
    def get_logs(self) -> pd.DataFrame:
        """Retrieve all logged anomalies"""
        return pd.DataFrame(self.log_data)

class SystemHealthMonitor:
    def __init__(self):
        self.metrics = []
        
    def record_metric(self, timestamp: datetime, cpu_usage: float, 
                     memory_usage: float, processing_time: float) -> None:
        """Record system health metrics"""
        metric = {
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'processing_time': processing_time
        }
        self.metrics.append(metric)
    
    def get_health_metrics(self) -> pd.DataFrame:
        """Get all recorded health metrics"""
        return pd.DataFrame(self.metrics)
    
    def get_system_status(self) -> Dict[str, float]:
        """Get current system status summary"""
        if not self.metrics:
            return {}
            
        df = pd.DataFrame(self.metrics)
        recent = df.iloc[-10:]  # Last 10 readings
        
        return {
            'avg_cpu': recent['cpu_usage'].mean(),
            'avg_memory': recent['memory_usage'].mean(),
            'avg_processing_time': recent['processing_time'].mean()
        }
