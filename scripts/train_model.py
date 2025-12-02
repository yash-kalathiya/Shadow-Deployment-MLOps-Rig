#!/usr/bin/env python3
"""
Model Training Script

Trains a challenger model for churn prediction and logs to model registry.
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Simple churn prediction model for demonstration."""
    
    def __init__(self, version: str):
        self.version = version
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.trained_at = None
        self.metrics = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list) -> "ChurnPredictor":
        """Train the model using simple logistic regression-like approach."""
        self.feature_names = feature_names
        self.trained_at = datetime.utcnow().isoformat()
        
        # Simple weight initialization based on feature correlation with target
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        
        for i in range(n_features):
            correlation = np.corrcoef(X[:, i], y)[0, 1]
            self.weights[i] = correlation * 0.5 if not np.isnan(correlation) else 0.0
        
        # Adjust bias based on class imbalance
        self.bias = np.log(y.mean() / (1 - y.mean() + 1e-10))
        
        logger.info(f"Model trained with {n_features} features")
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict churn probability."""
        raw_score = np.dot(X, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-raw_score))
        return probabilities
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict churn binary outcome."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y)
        
        # Precision, Recall, F1
        tp = np.sum((y_pred == 1) & (y == 1))
        fp = np.sum((y_pred == 1) & (y == 0))
        fn = np.sum((y_pred == 0) & (y == 1))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Simple AUC approximation
        auc = self._calculate_auc(y, y_proba)
        
        self.metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }
        
        return self.metrics
    
    def _calculate_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calculate AUC-ROC."""
        # Sort by score
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate AUC using trapezoidal rule
        tpr_list = []
        fpr_list = []
        
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        tp, fp = 0, 0
        for y in y_true_sorted:
            if y == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / (n_pos + 1e-10))
            fpr_list.append(fp / (n_neg + 1e-10))
        
        # Trapezoidal integration
        auc = np.trapz(tpr_list, fpr_list)
        return abs(auc)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ChurnPredictor":
        """Load model from file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare features for training."""
    feature_columns = [
        "days_since_last_login",
        "login_frequency_30d",
        "session_duration_avg",
        "total_transactions_90d",
        "transaction_value_avg",
        "support_tickets_30d",
        "subscription_tenure_days",
        "satisfaction_score",
    ]
    
    X = df[feature_columns].values
    y = df["churned"].values
    
    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_normalized = (X - X_mean) / X_std
    
    return X_normalized, y, feature_columns


def train_model(
    data_path: str,
    output_path: str,
    version: str,
) -> Dict[str, Any]:
    """Train a new challenger model."""
    logger.info(f"Loading training data from {data_path}")
    
    # Load data
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = ChurnPredictor(version=version)
    model.fit(X_train, y_train, feature_names)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Model metrics: {json.dumps(metrics, indent=2)}")
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    return {
        "version": version,
        "metrics": metrics,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "trained_at": model.trained_at,
    }


def main():
    parser = argparse.ArgumentParser(description="Train challenger model")
    parser.add_argument("--data", type=str, required=True, help="Training data parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output model file path")
    parser.add_argument("--version", type=str, required=True, help="Model version")
    
    args = parser.parse_args()
    
    results = train_model(
        data_path=args.data,
        output_path=args.output,
        version=args.version,
    )
    
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
