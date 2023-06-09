import logging
import os,sys
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from store.logger import logger
from store.exception import CustomException
from store.utils import load_numpy_array_data, load_object, save_object
from store.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from store.entity.config_entity import ModelTrainerConfig
from store.ml.metric.regression_metric import get_regression_score
from store.ml.model.estimator import StoreModel

class ModelTrainer:
    def __init__(self,
        data_transformation_artifact:DataTransformationArtifact,
        model_trainer_config:ModelTrainerConfig):
        try:
            logger.info(f"{'>>' * 20} Starting Model trainer {'<<' * 20}")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def train_model(self,x_train,y_train,params):
        try:
            gbr_reg = GradientBoostingRegressor()
            gbr_reg.set_params(**params)
            gbr_reg.fit(x_train,y_train)
            return gbr_reg
        except Exception as e:
            raise CustomException(e,sys)
        
    def perform_hyperparameter_tuning(self,x_train,y_train)->dict:
        params={          
            # 'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
            # 'criterion':['friedman_mse', 'squared_error'],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators':[100,200,300],
            'subsample':[0.6,0.7,0.8],
            'alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.99],
            # 'max_depth': [6,8,10],
            'learning_rate': [0.01, 0.05, 0.1],
        }
        rs = RandomizedSearchCV(GradientBoostingRegressor(), params, cv=3,n_jobs=-1)
        rs.fit(x_train,y_train)
        return rs.best_params_

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            params = self.perform_hyperparameter_tuning(x_train=x_train,y_train=y_train)
            logger.info(f'Best Hyperparameters: {params}')
            model = self.train_model(x_train=x_train,y_train=y_train,params=params)
            y_train_pred = model.predict(x_train)
            regression_train_metric = get_regression_score(y_true=y_train,y_pred=y_train_pred)
            logger.info(f'Training Metric: {regression_train_metric}')

            if regression_train_metric.r2_score <= self.model_trainer_config.expected_accuracy:
                raise Exception('Trained model is not good to provide expected accuracy')
            
            y_test_pred = model.predict(x_test)
            regression_test_metric = get_regression_score(y_true=y_test,y_pred=y_test_pred)
            logger.info(f'Testing Metric: {regression_test_metric}')

            # check overfitting and underfitting
            diff = abs(regression_train_metric.r2_score - regression_test_metric.r2_score)
            logger.info(f'Diff: {diff}')

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception('model is not good try to do more experimentation')
            
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            store_model = StoreModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=store_model)

            # model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric,
            )
            logger.info(f'Model trainer artifact: {model_trainer_artifact}')
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)

