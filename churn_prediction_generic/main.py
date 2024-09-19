import pandas as pd
import numpy as np
from churn_analysis import data_load, perform_eda
from model import train_model
filepath = input("Enter the filepath of csv")

preprocess_data = data_load(filepath)
preprocess_data['churn'] = preprocess_data['churn'].fillna(0).astype(int)  # Fill missing values and convert to int

perform_eda(preprocess_data)
trained_model = train_model(preprocess_data)
