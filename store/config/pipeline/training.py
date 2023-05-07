from store.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from store.exception import CustomException
from store.logger import logger
from store.entity.artifact_entity import DataIngestionArtifact
from store.constant.training_pipeline_config import *
from store.constant import TIMESTAMP
import sys
import os

class StoreConfig():
    def __init__(self,pipeline_name=PIPELINE_NAME,timestamp=TIMESTAMP):
        '''
        Author: Vijaykumar Rewate
        '''
        self.pipeline_name = pipeline_name
        self.timestamp = timestamp
        self.pipeline_config = self.get_pipeline_config()

    def get_pipeline_config(self)->TrainingPipelineConfig:
        ''''
        This function will provide pipeline configuration
        returns > PipelineConfig = namedtuple("PipelineConfig", ["pipeline_name", "artifact_dir"])
        '''
        try:
            artifact_dir = ARTIFACT_DIR
            pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir, pipeline_name=self.pipeline_name)
            logger.info(f'Pipeline configuration: {pipeline_config}')
            return pipeline_config
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_ingestion_config(self)->DataIngestionConfig:
        data_ingestion_dir = os.path.join(self.pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
        feature_store_file_path= os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR)
        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir=data_ingestion_dir,
            feature_store_file_path= os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME),
            training_file_path=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME),
            testing_file_path=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME),
            train_test_split_ratio=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
            collection_name=DATA_INGESTION_COLLECTION_NAME,
        )
        logger.info(f'Data Ingestion Config: {data_ingestion_config}')
        return data_ingestion_config


    