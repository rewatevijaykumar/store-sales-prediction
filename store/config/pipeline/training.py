from store.constant.training_pipeline_config.data_validation import DATA_VALIDATION_DIR_NAME, DATA_VALIDATION_DRIFT_REPORT_DIR, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME, DATA_VALIDATION_INVALID_DIR, DATA_VALIDATION_VALID_DIR
from store.entity.config_entity import DataValidationConfig, TrainingPipelineConfig, DataIngestionConfig
from store.exception import CustomException
from store.logger import logger
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
            artifact_dir = os.path.join(ARTIFACT_DIR,self.timestamp)
            pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir, pipeline_name=self.pipeline_name)
            logger.info(f'Pipeline configuration: {pipeline_config}')
            return pipeline_config
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            data_ingestion_dir = os.path.join(self.pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
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
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_validation_config(self)->DataValidationConfig:
        try:
            data_validation_dir = os.path.join(self.pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
            valid_data_dir=os.path.join(data_validation_dir,DATA_VALIDATION_VALID_DIR)
            invalid_data_dir=os.path.join(data_validation_dir,DATA_VALIDATION_INVALID_DIR)
            data_validation_config = DataValidationConfig(
                data_validation_dir=data_validation_dir,
                valid_data_dir=valid_data_dir,
                invalid_data_dir=invalid_data_dir,
                valid_train_file_path=os.path.join(valid_data_dir,TRAIN_FILE_NAME),
                valid_test_file_path=os.path.join(valid_data_dir,TEST_FILE_NAME),
                invalid_train_file_path=os.path.join(invalid_data_dir,TRAIN_FILE_NAME),
                invalid_test_file_path=os.path.join(invalid_data_dir,TEST_FILE_NAME),
                drift_report_file_path=os.path.join(data_validation_dir,
                                                    DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
            )
            logger.info(f"Data validation config: {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise CustomException(e,sys)


    