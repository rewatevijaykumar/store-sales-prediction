import os
from re import A
import sys
from store.components.training.data_ingestion import DataIngestion
from store.entity.artifact_entity import DataIngestionArtifact
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
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys)