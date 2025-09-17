import shap
import joblib
import pandas as pd

# Load model and data
model = joblib.load('models/xgboost_model.pkl')
df = pd.read_csv('data/creditcard.csv')
X = df.drop('Class', axis=1)

# SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Plot
shap.summary_plot(shap_values, X, show=False)
