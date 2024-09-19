import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score


import joblib

def load_trained_model():
    """
    Loads a pre-trained model from disk. Modify the file path as per your setup.
    """
    return joblib.load('trained_model.pkl')  # Assuming the model was saved as 'trained_model.pkl'


def save_model(model, model_name: str):
    joblib.dump(model, f'{model_name}_model.pkl')


def train_model(data: pd.DataFrame, model_name: str = 'random_forest') -> any:
    X = data.drop('churn', axis=1)
    y = data['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if model_name == 'random_forest':
        model = RandomForestClassifier()
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}, AUC-ROC: {roc_auc}, Precision: {precision}, Recall: {recall}")

    save_model(model, model_name)  # Save the model with its configuration

    return model

import shap

def explain_with_shap(data: pd.DataFrame, model) -> dict:
    """
    Generate SHAP values for a given dataset.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    
    # Summarize SHAP values for visualization
    return {
        "shap_values": shap_values.values.tolist(),  # Convert to list for serialization
        "feature_names": data.columns.tolist()
    }

from lime.lime_tabular import LimeTabularExplainer

def explain_with_lime(data: pd.DataFrame, model) -> dict:
    """
    Generate LIME explanations for a given dataset.
    """
    explainer = LimeTabularExplainer(
        training_data=data.values,
        feature_names=data.columns.tolist(),
        class_names=['Not Churn', 'Churn'],
        mode='classification'
    )

    # Assuming we're explaining a single instance
    instance = data.iloc[0]
    explanation = explainer.explain_instance(instance.values, model.predict_proba)
    
    # Convert explanation to a dictionary format
    return explanation.as_list()
