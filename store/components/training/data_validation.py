import os
import sys
from pandas import DataFrame
from store.constant.training_pipeline_config import SCHEMA_FILE_PATH
from store.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from store.entity.config_entity import DataValidationConfig
from store.exception import CustomException
from store.logger import logger
from store.utils import read_yaml_file, write_yaml_file, read_data
import pandas as pd
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            logger.info(f"{'>>' * 20} Starting data validation {'<<' * 20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_number_of_columns(self,dataframe:DataFrame)->bool:
        try:
            number_of_columns = self._schema_config['columns']
            if len(dataframe.columns) == len(number_of_columns):
                return True
            else:
                return False
        except Exception as e:
            raise CustomException(e,sys)
        
    def is_numerical_column_exists(self,dataframe:DataFrame)->bool:
        try:
            numerical_columns = self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns
            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)
            logger.info(f'Missing numerical columns: {missing_numerical_columns}')
            return numerical_column_present
        except Exception as e:
            raise CustomException(e,sys)
                
    def detect_dataset_drift(self,base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report ={}
            numerical_columns = self._schema_config['numerical_columns']
            for column in numerical_columns:
                d1 = base_df[column]
                d2  = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <= is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message = ''

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # reading data from train and test file location
            train_dataframe = read_data(train_file_path)
            test_dataframe = read_data(test_file_path)

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f'{error_message} Train dataset does not contain all columns \n'
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f'{error_message} Test dataset does not contain all columns \n'

            # Validate numeric columns
            status = self.is_numerical_column_exists(dataframe=train_dataframe)
            if not status:
                error_message = f'{error_message} Train dataset does not contain all numerical columns\n'
            status = self.is_numerical_column_exists(dataframe=test_dataframe)
            if not status:
                error_message = f'{error_message} Test dataset does not contain all numerical columns\n'


            if len(error_message) > 0:
                raise Exception(error_message)
            
            # check data drift
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                # invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                # invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logger.info(f'Data Validation Artifact: {data_validation_artifact}')
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)