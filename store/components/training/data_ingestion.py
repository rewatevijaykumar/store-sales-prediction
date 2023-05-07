import os
import sys
from store.constant.training_pipeline_config import SCHEMA_FILE_PATH
from store.entity.config_entity import DataIngestionConfig
from store.entity.artifact_entity import DataIngestionArtifact
from store.logger import logger
from store.exception import CustomException
from store.utils import read_yaml_file
from pandas import DataFrame
from store.data_access.store_data import StoreData
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logger.info(f"{'>>' * 20} Starting data ingestion {'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_feature_store(self)->DataFrame:
        """
        Export mongodb collection record as dataframe into feature store
        return dataframe
        """
        try:
            logger.info('Exporting data into feature store from mongodb')
            store_data = StoreData()
            dataframe = store_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            #create folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
    
    def split_data_as_train_test(self, dataframe:DataFrame)->None:
        '''
        Feature store dataset will be split into train and test file
        '''
        try:
            
            train_set, test_set = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logger.info("Performed train test split on DataFrame")

            logger.info("Exited train test split method of Data Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logger.info("Exporting train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logger.info('Exported tran and test file path')


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            logger.info('Starting Data Ingestion')
            dataframe = self.export_data_into_feature_store()
            
            # dataframe.drop(self._schema_config['drop_columns'],axis=1)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            logger.info(f'Data Ingestion artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)
