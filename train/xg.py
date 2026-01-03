"""XGBoost model definition"""

# Cut/Hold/Hike

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example: X = features, y = target (0=cut, 1=hold, 2=hike)
X = pd.DataFrame(...)  # shape (n_samples, n_features)
y = pd.Series(...)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost classifier
model = xgb.XGBClassifier(
    objective='multi:softprob',  # multi-class
    num_class=3,                 # cut/hold/hike
    max_depth=3,                 # small tree for simplicity
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss'       # multi-class log-loss
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy:", accuracy)
