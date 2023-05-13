import argparse

from fastapi import FastAPI
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
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, Response
from uvicorn import run as app_run
from store.constant.application import APP_HOST, APP_PORT
import pandas as pd
import numpy as np
from store.utils import load_object, read_yaml_file
from sklearn.preprocessing import QuantileTransformer
import re
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/',tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def train_route():
    try:
        training_pipeline = TrainingPipeline(store_config=StoreConfig())
        if training_pipeline.is_pipeline_running:
            return Response('Training pipeline is already running')
        training_pipeline.run_pipeline()
    except Exception as e:
        return Response(f'Error occured! {e}')
    

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file.file, na_values = 'na')
        df.replace({'na':np.nan}, inplace=True)
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
            return Response('Model is not available')
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        target_preprocessor_path = model_resolver.get_target_preprocessor()
        target_preprocessor = load_object(file_path=target_preprocessor_path)
        df['predicted_column'] = target_preprocessor.inverse_transform(df['predicted_column'].to_numpy().reshape(-1,1))
        # Return the DataFrame as a JSON response
        response = df['predicted_column'].to_json(orient="records")

        return response

    except Exception as e:
        print(CustomException(e,sys))
        return Response(f'Error occured! {e}')

def main():
    try:
        training_pipeline = TrainingPipeline(store_config=StoreConfig())
        if training_pipeline.is_pipeline_running:
            raise Exception('Training pipeline is already running')
        training_pipeline.run_pipeline()
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    try:
        main()

        app_run(app,host=APP_HOST, port=APP_PORT)
    except Exception as e:
        raise e