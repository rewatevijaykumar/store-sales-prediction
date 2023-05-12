import os
from re import A
import sys
from store.components.training.data_ingestion import DataIngestion
from store.components.training.data_transformation import DataTransformation
from store.components.training.data_validation import DataValidation
from store.components.training.model_evaluation import ModelEvaluation
from store.components.training.model_pusher import ModelPusher
from store.components.training.model_trainer import ModelTrainer
from store.constant.training_pipeline_config import data_ingestion, model_evaluation
from store.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelEvaluationArtifact, ModelTrainerArtifact
from store.entity.config_entity import ModelPusherConfig
from store.exception import CustomException
from store.logger import logger
from store.config.pipeline.training import StoreConfig

class TrainingPipeline:
    is_pipeline_running = False
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

    def start_model_evaluation(self, 
                               data_validation_artifact:DataValidationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_evaluation_config = self.store_config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
                model_evaluation_config=model_evaluation_config
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e,sys)

 
    def start_model_pusher(self,
                           data_transformation_artifact:DataTransformationArtifact,
                            model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = self.store_config.get_model_pusher_config()
            model_pusher = ModelPusher(
                data_transformation_artifact=data_transformation_artifact,
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        try:
            TrainingPipeline.is_pipeline_running=True
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                raise Exception('Trained model is not better than the best model')
            model_pusher_artifact = self.start_model_pusher(
                data_transformation_artifact=data_transformation_artifact,
                model_evaluation_artifact=model_evaluation_artifact)
            TrainingPipeline.is_pipeline_running=False
        except Exception as e:
            TrainingPipeline.is_pipeline_running=False
            raise CustomException(e,sys)