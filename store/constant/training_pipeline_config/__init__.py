import os
from store.constant.training_pipeline_config.data_ingestion import *

SAVED_MODEL_DIR = os.path.join('saved_models')


# define common constant variable for training pipeline

TARGET_COLUMN = 'Item_Outlet_Sales'
PIPELINE_NAME:str = 'store'
ARTIFACT_DIR: str = os.path.join(os.getcwd(), "artifact")
FILE_NAME:str = 'store.csv'

TRAIN_FILE_NAME:str = 'train.csv'
TEST_FILE_NAME:str = 'test.csv'

PREPROCESSING_OBJECT_FILE_NAME:str = 'preprocessing.pkl'
MODEL_FILE_NAME:str = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join('config', 'schema.yaml')
SCHEMA_DROP_COLS = 'drop_columns'
