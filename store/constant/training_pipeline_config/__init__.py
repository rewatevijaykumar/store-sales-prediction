import os
from store.constant.training_pipeline_config.data_ingestion import *
from store.constant.training_pipeline_config.data_validation import *
from store.constant.training_pipeline_config.data_transformation import *
from store.constant.training_pipeline_config.model_trainer import *
from store.constant.training_pipeline_config.model_evaluation import *
from store.constant.training_pipeline_config.model_pusher import *

SAVED_MODEL_DIR = os.path.join('saved_models')


# define common constant variable for training pipeline

TARGET_COLUMN = 'Item_Outlet_Sales'
PIPELINE_NAME:str = 'store'
ARTIFACT_DIR: str = os.path.join(os.getcwd(), "artifact")
FILE_NAME:str = 'store.csv'

TRAIN_FILE_NAME:str = 'train.csv'
TEST_FILE_NAME:str = 'test.csv'

PREPROCESSING_OBJECT_FILE_NAME:str = 'preprocessing.pkl'
TARGET_PREPROCESSING_OBJECT_FILE_NAME:str = 'target_preprocessing.pkl'
MODEL_FILE_NAME:str = 'model.pkl'
SCHEMA_FILE_PATH = os.path.join('schema_config', 'schema.yaml')
SCHEMA_DROP_COLS = 'drop_columns'
