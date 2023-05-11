import os
from re import A
import sys
from store.components.training.data_ingestion import DataIngestion
from store.components.training.data_transformation import DataTransformation
from store.components.training.data_validation import DataValidation
from store.components.training.model_trainer import ModelTrainer
from store.constant.training_pipeline_config import data_ingestion
from store.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from store.exception import CustomException
from store.logger import logger
from store.config.pipeline.training import StoreConfig

class TrainingPipeline:
    def __init__(self,store_config:StoreConfig):
        self.store_config = store_config

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion_config = self.store_config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = self.store_config.get_data_validation_config()
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = self.store_config.get_data_transformation_config()
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            CustomException(e,sys)

    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = self.store_config.get_model_trainer_config()
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        except Exception as e:
            raise CustomException(e,sys)