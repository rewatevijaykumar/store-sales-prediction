import os
import sys
from category_encoders import OneHotEncoder, OrdinalEncoder
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from store.constant.training_pipeline_config import SCHEMA_FILE_PATH, TARGET_COLUMN
from store.logger import logger
from store.exception import CustomException
from store.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from store.config.pipeline.training import DataTransformationConfig
from store.utils import read_data, read_yaml_file, save_numpy_array_data,save_object
import numpy as np

class DataTransformation:
    def __init__(self, 
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig)->DataTransformationArtifact:
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            logger.info(f"{'>>' * 20} Starting data transformation {'<<' * 20}")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)

    def check_unique_values_in_columns(self,dataframe:DataFrame)->DataFrame:
        try:
            dataframe['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg':'Regular'}, inplace=True)
            return dataframe
        except Exception as e:
            CustomException(e,sys)

    def check_data_type_of_columns(self,dataframe:DataFrame)->DataFrame:
        try:
            columns = self._schema_config['columns']
            # Create a dictionary to map column names to their corresponding data types
            data_types = {}
            for col in columns:
                data_types.update(col)
            # Convert the columns in the DataFrame to their respective data types
            dataframe = dataframe.astype(data_types)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)

    def get_target_transformation_object(self)->Pipeline:
        try:
            target_preprocessor = Pipeline(steps=[
                ('scaler', QuantileTransformer(output_distribution='normal'))
            ])
            return target_preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_transformation_object(self,)->Pipeline:
        try:
            num_features = self._schema_config['numeric_columns']
            ohe_features = self._schema_config['onehot_columns']
            ord_features = self._schema_config['ordinal_columns']
            # # # Fit simple imputer with strategy median

            ## Create preprocessing pipelines for each datatype 
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', QuantileTransformer(output_distribution='normal'))])

            onehot_transformer = Pipeline(steps=[
                ('ohe', OneHotEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', QuantileTransformer(output_distribution='normal'))])
            
            ordinal_transformer = Pipeline(steps=[
                ('ord', OrdinalEncoder()),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', QuantileTransformer(output_distribution='normal'))])

            ## Putting the preprocessing steps together
            preprocessor = ColumnTransformer([
                    ('numerical', numerical_transformer, num_features),
                    ('ordinal', ordinal_transformer, ord_features),
                    ('categorical', onehot_transformer, ohe_features)],
                    remainder='passthrough')


            ## Create example pipeline with kNN as estimator
            quant_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
            ])
            return quant_pipeline
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")
            train_df = read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = read_data(self.data_validation_artifact.valid_test_file_path)
            train_df = self.check_data_type_of_columns(train_df)
            train_df = self.check_unique_values_in_columns(train_df)
            test_df = self.check_unique_values_in_columns(test_df)
            test_df = self.check_data_type_of_columns(test_df)
            preprocessor = self.get_data_transformation_object()
            target_preprocessor = self.get_target_transformation_object()

            #training dataframe
            input_feature_train_df = train_df.drop([TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            #testing dataframe
            input_feature_test_df = test_df.drop([TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train_df = preprocessor_object.transform(input_feature_train_df)
            transformed_input_feature_test_df = preprocessor_object.transform(input_feature_test_df)

            transformed_target_feature_train_df = target_preprocessor.fit_transform(target_feature_train_df.to_numpy().reshape(-1,1))
            transformed_target_feature_test_df = target_preprocessor.fit_transform(target_feature_test_df.to_numpy().reshape(-1,1))
            
            train_arr = np.c_[transformed_input_feature_train_df,np.array(transformed_target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_df,np.array(transformed_target_feature_test_df)]

            # save numpy array data
            logger.info("Saving numpy array data and preprocessor object")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object(self.data_transformation_config.transformed_target_object_file_path, target_preprocessor)

            # preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_target_object_file_path=self.data_transformation_config.transformed_target_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logger.info(f'Data transformation artifact: {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    

