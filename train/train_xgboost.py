"""
Train XGBoost Interest Rate Classifier

Loads data from data/interest/processed/Interest_Features.parquet
Trains XGBoost with Focal Loss and SMOTE
Saves model to models/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .xg import InterestRateClassifier


def load_data():
    """Load and prepare interest rate features."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "interest" / "processed" / "Interest_Features.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run: python -m data.interest.collector && python -m data.interest.formatter"
        )
    
    df = pd.read_parquet(data_path)
    
    # Features (exclude date and target)
    feature_cols = [
        'Core_PCE_1M_Ann',
        'Unemployment_Gap',
        'CPI_Surprise_Proxy',
        'Spread_3M_2Y',
        'Spread_2Y_10Y',
        'Fin_Cond_Ind',
        'DFEDTARU'
    ]
    
    X = df[feature_cols].values
    y = df['Fed_Target'].values  # 'cut', 'hold', 'hike'
    dates = df['date'].values
    
    return X, y, dates, feature_cols


def time_aware_split(X, y, dates, test_size=0.2):
    """
    Split data preserving time order (no future leakage).
    Uses last `test_size` fraction as test set.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test


def train():
    """Main training function."""
    print("Loading data...")
    X, y, dates, feature_cols = load_data()
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {feature_cols}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt} ({100*cnt/len(y):.1f}%)")
    
    # Time-aware split
    X_train, X_test, y_train, y_test, dates_train, dates_test = time_aware_split(X, y, dates)
    
    print(f"\nTrain: {len(X_train)} samples ({dates_train[0]} to {dates_train[-1]})")
    print(f"Test:  {len(X_test)} samples ({dates_test[0]} to {dates_test[-1]})")
    
    # Train
    print("\nTraining XGBoost with Focal Loss...")
    model = InterestRateClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        gamma_focal=2.0,
        use_smote=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\nMacro F1 Score: {f1_macro:.4f}")
    
    # Save model
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "interest_rate_classifier.json"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    return model, f1_macro


if __name__ == "__main__":
    train()
