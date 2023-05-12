from store.exception import CustomException
from store.logger import logger
from store.entity.artifact_entity import DataTransformationArtifact, ModelEvaluationArtifact, ModelPusherArtifact, ModelTrainerArtifact, DataValidationArtifact
from store.entity.config_entity import ModelPusherConfig
import os,sys
import shutil

class ModelPusher:
    def __init__(self, 
        model_pusher_config:ModelPusherConfig, 
        data_transformation_artifact:DataTransformationArtifact,
        model_evaluation_artifact:ModelEvaluationArtifact):
        try:
            logger.info(f"{'>>' * 20} Starting model pusher {'<<' * 20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_model_pusher(self)-> ModelPusherArtifact:
        try:
            trained_model_path = self.model_evaluation_artifact.trained_model_path
            target_object = self.data_transformation_artifact.transformed_target_object_file_path

            #create model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path,dst=model_file_path)

            # saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            target_object_file_path = self.model_pusher_config.target_object_file_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path,dst=saved_model_path)
            shutil.copy(src=target_object,dst=target_object_file_path)

            # prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path,
                                                        model_file_path=model_file_path,
                                                        target_object_file_path=target_object_file_path)
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e,sys)