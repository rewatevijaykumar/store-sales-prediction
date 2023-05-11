from store.entity.config_entity import DataTransformationConfig, DataValidationConfig, ModelTrainerConfig, TrainingPipelineConfig, DataIngestionConfig
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

    def get_data_transformation_config(self)->DataTransformationConfig:
        try:
            data_transformation_dir = os.path.join(self.pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
            data_transformation_config = DataTransformationConfig(
                data_transformation_dir=data_transformation_dir,
                transformed_train_file_path=os.path.join(data_transformation_dir, 
                                                         DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                         TRAIN_FILE_NAME.replace('csv', 'npy')),
                transformed_test_file_path=os.path.join(data_transformation_dir,
                                                        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                        TEST_FILE_NAME.replace('csv', 'npy')),
                transformed_object_file_path=os.path.join(data_transformation_dir,
                                                          DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                          PREPROCESSING_OBJECT_FILE_NAME),
                transformed_target_object_file_path=os.path.join(data_transformation_dir,
                                                          DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                          TARGET_PREPROCESSING_OBJECT_FILE_NAME),
            )
            logger.info(f'Data Transformation Config: {data_transformation_config}')
            return data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_model_trainer_config(self)->ModelTrainerConfig:
        try:
            model_trainer_dir = os.path.join(self.pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME, self.timestamp)
            model_trainer_config = ModelTrainerConfig(
                model_trainer_dir=model_trainer_dir,
                trained_model_file_path=os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME),
                expected_accuracy=MODEL_TRAINER_EXPECTED_SCORE,
                overfitting_underfitting_threshold=MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
                 
            )
            logger.info(f'Model Trainer Config: {model_trainer_config}')
            return model_trainer_config
        except Exception as e:
            raise CustomException(e,sys)


    