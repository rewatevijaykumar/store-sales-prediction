import traceback
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json

import argparse

from store.config.pipeline.training import StoreConfig
from store.constant.training_pipeline_config import SAVED_MODEL_DIR, SCHEMA_FILE_PATH
from store.data_access.store_data import StoreData
from store.exception import CustomException
from store.logger import logger
import os,sys
from store.config.mongo_db_connection import MongoDBClient
from store.pipeline import training_pipeline
from store.pipeline.training_pipeline import TrainingPipeline
from store.ml.model.estimator import ModelResolver
from store.constant.application import APP_HOST, APP_PORT
import pandas as pd
import numpy as np
from store.utils import load_object, read_yaml_file
from sklearn.preprocessing import QuantileTransformer
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        training_pipeline = TrainingPipeline(store_config=StoreConfig())
        # Check if the pipeline is already running
        if training_pipeline.is_pipeline_running:
            message = 'Training pipeline is already running'
            return jsonify(message=message)
        # Start the pipeline
        training_pipeline.run_pipeline()
        message = 'Training pipeline completed successfully'
        return jsonify(message=message)
    except CustomException as e:
        traceback.print_exc()  # Print the traceback
        error_message = re.search(r'error message: \[(.*?)\]', str(e)).group(1)
        return jsonify(error=error_message)        

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the form data
        data = request.get_json()
        item_identifier = data['itemIdentifier']
        item_weight = float(data['itemWeight'])
        item_fat_content = data['itemFatContent']
        item_visibility = float(data['itemVisibility'])
        item_type = data['itemType']
        item_mrp = float(data['itemMRP'])
        outlet_identifier = data['outletIdentifier']
        outlet_year = int(data['outletYear'])
        outlet_size = data['outletSize']
        outlet_location = data['outletLocation']
        outlet_type = data['outletType']
        item_sales = float(data['itemSales'])

        # Create a dictionary with the form data
        form_data = {
            'Item_Identifier': item_identifier,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location,
            'Outlet_Type': outlet_type,
            'Item_Outlet_Sales': item_sales
        }

        # Create a DataFrame from the JSON data
        df = pd.DataFrame([form_data])
        columns = read_yaml_file(SCHEMA_FILE_PATH)['columns'][:-1]
        # Create a dictionary to map column names to their corresponding data types
        data_types = {}
        for col in columns:
            data_types.update(col)
        # Convert the columns in the DataFrame to their respective data types
        df = df.astype(data_types)
        df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg':'Regular'}, inplace=True)
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return jsonify('Model is not available')
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        target_preprocessor_path = model_resolver.get_target_preprocessor()
        target_preprocessor = load_object(file_path=target_preprocessor_path)
        df['predicted_column'] = target_preprocessor.inverse_transform(df['predicted_column'].to_numpy().reshape(-1,1))
        # Return the DataFrame as a JSON response
        results = df['predicted_column'].round(decimals=2).to_json(orient="records")
        return results
    except Exception as e:
        traceback.print_exc()  # Print the traceback
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    try:
        app.run(host=APP_HOST, port=APP_PORT,debug=True)
    except Exception as e:
        raise e