
# 🛡️Credit Card Fraud Detection
### A machine learning project to detect fraudulent credit card transactions using supervised learning techniques. This repository explores data preprocessing, model training, evaluation, and interpretability using SHAP and LIME.
## Project Structure

credit-card-fraud-detection/   
├── data/                  # Raw dataset (creditcard.csv)  
├── notebooks/             # Jupyter notebooks for each stage  
├── scripts/               # Modular Python scripts  
├── models/                # Saved model files (.pkl)  
├── plots/                 # SHAP & LIME visualizations  
├── requirements.txt       # Python dependencies  
└── README.md              # Project overview
## Dataset

- Source: Kaggle Credit Card Fraud Dataset[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud] (Dataset is available in the repository also.)
- Description: Contains transactions made by European cardholders in September 2013.
- Size: 284,807 transactions with 492 frauds (highly imbalanced)
- Features: 30 columns including anonymized PCA components (V1 to V28), Time, Amount, and Class (target)

## Objectives

To build a robust classification model that can:
- Accurately detect fraudulent transactions
- Handle class imbalance effectively
- Provide interpretable insights using SHAP and LIME

## Models Used

- Logistic Regression : Baseline model with good interpretability.
- Random Forest       :  Ensemble model with warm start tuning.
- XGBoost : Gradient boosting with hyperparameter tuning

## Preprocessing Step

- Feature scaling using StandardScaler
- Handling class imbalance with SMOTE
- Train-test split (80/20)
- Optional PCA visualization for feature space

## Evaluation Metrics

- Precision, Recall, F1-Score
- Confusion Matrix
- Area Under Precision-Recall Curve (AUPRC)
- ROC Curve (optional)

## Model Interpretability

SHAP (SHapley Additive exPlanations)
- Global feature importance
- Dependence plots
- Force plots for individual predictions
LIME (Local Interpretable Model-Agnostic Explanations)
- Local explanations for selected transactions
- Visual breakdown of feature contributions

## How to Run

1. Clone the repository

```bash
  git clone https://github.com/hitha0511/Credit-card-Fraud-Detection-data-viz-
cd credit-card-fraud-detection
```
2. Install dependencies
```bash
  pip install -r requirements.txt
```
3. Run notebooks
Open each notebook in order:
- 01_data_loading.ipynb
- 02_preprocessing.ipynb
- 03_model_training.ipynb
- 04_model_evaluation.ipynb
- 05_shap_lime_interpretation.ipynb

4. Run scripts (optional)
```bash
  python scripts/train_xgboost.py
  python scripts/interpret_model.py
```
## Key Results

- XGBoost achieved highest AUPRC and recall
- SHAP revealed V14, V10, and Amount as top predictors
- LIME confirmed model decisions on edge cases

##  Learnings
- Importance of handling class imbalance
- Trade-offs between precision and recall in fraud detection
- Value of model interpretability in real-world applications

## Author

Hitha
Aspiring Software Engineer | Java, Python, SQL | Passionate about algorithmic logic and digital inclusion.

📫 LinkedIn: https://www.linkedin.com/in/hitharn0504/ | 📧 hitharn0504@gmail.com

📜 License
This project is licensed under the [MIT] License. See the LICENSE file for details.

Would you like me to generate the actual requirements.txt or any of the scripts next? Or help you write a LinkedIn post to showcase this project?

## License
This project is licensed under the MIT License. See the LICENSE file for details
[MIT](https://github.com/hitha0511/Credit-card-Fraud-Detection-data-viz-/new/main)

