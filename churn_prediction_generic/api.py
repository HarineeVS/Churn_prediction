from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from churn_analysis import data_load
from model import load_trained_model, explain_with_shap, explain_with_lime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

app = FastAPI()

# Model and Dataset management
models = {}
current_model_name = 'default_model'
current_dataset = None

class PredictionInput(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    
@app.on_event("startup")
async def setup():
    global models
    models[current_model_name] = load_trained_model(current_model_name)

@app.post("/load_dataset")
async def load_dataset(filepath: str, preprocess_method: str = 'standard'):
    global current_dataset
    current_dataset = data_load(filepath, preprocess_method)
    return {"status": "Dataset loaded and preprocessed"}


def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data using the same logic as in churn_analysis.py.
    This function ensures consistency between training and inference.
    """
    # Encoding categorical features
    for col in input_data.select_dtypes(include='object').columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])
        else:
            # If a label encoder for the column does not exist, we create and fit one
            le = LabelEncoder()
            input_data[col] = le.fit_transform(input_data[col])
            label_encoders[col] = le

    # Scaling features
    scaled_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
    
    return scaled_data

# Initialize label encoders and scaler
label_encoders = {}
scaler = StandardScaler()

@app.on_event("startup")
async def setup():
    global models, scaler, label_encoders
    scaler = StandardScaler()
    models[current_model_name] = load_trained_model(current_model_name)


@app.post("/train_model")
async def train_model(model_name: str = 'random_forest'):
    global models, current_dataset
    if current_dataset is None:
        return {"error": "No dataset loaded"}
    
    models[model_name] = train_model(current_dataset, model_name)
    return {"status": f"Model {model_name} trained"}

@app.post("/predict")
async def predict(data: PredictionInput):
    global models, current_model_name
    if current_model_name not in models:
        return {"error": "Model not loaded"}
    
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    processed_data = preprocess_input(input_data)
    prediction = models[current_model_name].predict(processed_data)
    return {"churn_prediction": int(prediction[0])}

@app.post("/explain/shap")
async def explain_shap(data: PredictionInput):
    global models, current_model_name
    if current_model_name not in models:
        return {"error": "Model not loaded"}
    
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    processed_data = preprocess_input(input_data)
    shap_explanation = explain_with_shap(processed_data, models[current_model_name])
    return shap_explanation

@app.post("/explain/lime")
async def explain_lime(data: PredictionInput):
    global models, current_model_name
    if current_model_name not in models:
        return {"error": "Model not loaded"}
    
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    processed_data = preprocess_input(input_data)
    lime_explanation = explain_with_lime(processed_data, models[current_model_name])
    return {"lime_explanation": lime_explanation}
