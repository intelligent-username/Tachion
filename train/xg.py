"""
XGBoost Interest Rate Classifier

Predicts Fed rate decisions: cut (0), hold (1), hike (2)
Uses Focal Loss for class imbalance and optional SMOTE for oversampling.
"""

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

# Focal Loss for XGBoost (handles class imbalance better than standard softmax)
# Focal Loss: FL(p) = -alpha * (1 - p)^gamma * log(p)


def focal_loss_objective(preds, dtrain, gamma=2.0, alpha=None):
    """
    Focal Loss gradient and hessian for XGBoost multi-class.
    
    :param preds: Raw predictions (before softmax), shape (n_samples * n_classes,)
    :param dtrain: DMatrix with labels
    :param gamma: Focusing parameter (higher = more focus on hard examples)
    :param alpha: Class weights (optional), shape (n_classes,)
    :return: grad, hess
    """
    labels = dtrain.get_label().astype(int)
    n_samples = len(labels)
    n_classes = 3  # cut, hold, hike
    
    # Reshape preds to (n_samples, n_classes)
    preds = preds.reshape(n_samples, n_classes)
    
    # Softmax
    preds_max = preds.max(axis=1, keepdims=True)
    exp_preds = np.exp(preds - preds_max)
    softmax = exp_preds / exp_preds.sum(axis=1, keepdims=True)
    
    # One-hot encode labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), labels] = 1
    
    # Alpha weights (default: equal)
    if alpha is None:
        alpha = np.ones(n_classes) / n_classes
    alpha = np.array(alpha).reshape(1, -1)
    
    # Focal loss gradient and hessian
    # p_t = softmax for true class
    p_t = (softmax * y_onehot).sum(axis=1, keepdims=True)
    p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
    
    # Gradient: d(FL)/d(z_k) 
    # For true class: -alpha * gamma * (1-p)^(gamma-1) * log(p) * p * (1-p) + alpha * (1-p)^gamma * (1-p)
    # Simplified focal gradient
    focal_weight = alpha * (1 - p_t) ** gamma
    
    grad = focal_weight * (softmax - y_onehot)
    
    # Hessian approximation (diagonal)
    hess = focal_weight * softmax * (1 - softmax)
    hess = np.maximum(hess, 1e-6)  # numerical stability
    
    return grad.flatten(), hess.flatten()


def focal_loss_eval(preds, dtrain, gamma=2.0):
    """
    Focal Loss evaluation metric for XGBoost.
    """
    labels = dtrain.get_label().astype(int)
    n_samples = len(labels)
    n_classes = 3
    
    preds = preds.reshape(n_samples, n_classes)
    
    # Softmax
    preds_max = preds.max(axis=1, keepdims=True)
    exp_preds = np.exp(preds - preds_max)
    softmax = exp_preds / exp_preds.sum(axis=1, keepdims=True)
    
    # Get probability for true class
    p_t = softmax[np.arange(n_samples), labels]
    p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
    
    # Focal loss: -sum((1 - p_t)^gamma * log(p_t)) / n
    fl = -np.mean((1 - p_t) ** gamma * np.log(p_t))
    
    return "focal_loss", fl


class InterestRateClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier for Fed interest rate decisions with Focal Loss.
    
    Target classes:
        0 = cut
        1 = hold  
        2 = hike
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        gamma_focal=2.0,
        use_smote=True,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gamma_focal = gamma_focal
        self.use_smote = use_smote
        self.random_state = random_state
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
    
    def _apply_smote(self, X, y):
        """
        Apply SMOTE oversampling to balance classes.
        Falls back to random oversampling if imblearn not available.
        """
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res
        except ImportError:
            # Manual random oversampling fallback
            unique, counts = np.unique(y, return_counts=True)
            max_count = counts.max()
            
            X_list, y_list = [X], [y]
            for cls, cnt in zip(unique, counts):
                if cnt < max_count:
                    idx = np.where(y == cls)[0]
                    oversample_idx = np.random.choice(idx, max_count - cnt, replace=True)
                    X_list.append(X[oversample_idx])
                    y_list.append(y[oversample_idx])
            
            return np.vstack(X_list), np.hstack(y_list)
    
    def fit(self, X, y):
        """
        Train the classifier.
        
        :param X: Features array (n_samples, n_features)
        :param y: Labels - can be strings ('cut', 'hold', 'hike') or integers
        :return: self
        """
        # Encode labels if strings
        if isinstance(y[0], str):
            y = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder.fit(['cut', 'hold', 'hike'])
        
        self.classes_ = self.label_encoder.classes_
        
        # Apply SMOTE if requested
        if self.use_smote:
            X, y = self._apply_smote(X, y)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # XGBoost params
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'seed': self.random_state,
            'disable_default_eval_metric': True,
        }
        
        # Custom objective with focal loss
        def obj(preds, dtrain):
            return focal_loss_objective(preds, dtrain, gamma=self.gamma_focal)
        
        def feval(preds, dtrain):
            return [focal_loss_eval(preds, dtrain, gamma=self.gamma_focal)]
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=obj,
            custom_metric=feval,
            verbose_eval=False
        )
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        :param X: Features array
        :return: Probabilities (n_samples, 3)
        """
        dtest = xgb.DMatrix(X)
        raw_preds = self.model.predict(dtest)
        
        # Apply softmax
        raw_preds = raw_preds.reshape(-1, 3)
        exp_preds = np.exp(raw_preds - raw_preds.max(axis=1, keepdims=True))
        probs = exp_preds / exp_preds.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels.
        
        :param X: Features array
        :return: Predicted labels (as original strings if fitted with strings)
        """
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(pred_idx)
    
    def save(self, path):
        """Save model to file."""
        self.model.save_model(path)
    
    def load(self, path):
        """Load model from file."""
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self
