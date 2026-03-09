import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Sample dataset
data = pd.DataFrame({
    "area": [1000, 1500, 2000, 2500, 3000],
    "price": [200000, 300000, 400000, 500000, 600000]
})

X = data[["area"]]
y = data["price"]

# ✅ Create Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
    ])

# Train
pipeline.fit(X, y)

# Save entire pipeline
joblib.dump(pipeline, "model.joblib")

print("Model saved successfully!")