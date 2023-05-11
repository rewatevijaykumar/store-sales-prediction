from store.constant.training_pipeline_config import TARGET_COLUMN
from store.exception import CustomException
from store.logger import logger
import os, sys
from store.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataValidationArtifact
from store.config.pipeline.training import ModelEvaluationConfig
import pandas as pd
from store.ml.model.estimator import ModelResolver
from store.utils import load_object, write_yaml_file
from store.ml.metric.regression_metric import get_regression_score

class ModelEvaluation:
    def __init__(self, 
                 model_trainer_artifact:ModelTrainerArtifact, 
                 data_validation_artifact:DataValidationArtifact,
                 model_evaluation_config:ModelEvaluationConfig):
        try:
            logger.info(f"{'>>' * 20} Starting model evaluation {'<<' * 20}")
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            valid_train_file_path = os.path.join(self.data_validation_artifact.valid_train_file_path)
            valid_test_file_path = os.path.join(self.data_validation_artifact.valid_test_file_path)

            # valid train and test file dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df, test_df])
            y_true = df[TARGET_COLUMN]
            df = df.drop(TARGET_COLUMN, axis=1)

            # loading trained model
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted = True
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    trained_model_path=train_model_file_path,
                    improved_accuracy=None,
                    best_model_path=None,
                    train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                    best_model_metric_artifact=None)
                logger.info(f'Model evaluation artifact: {model_evaluation_artifact}')
                return model_evaluation_artifact
            
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_regression_score(y_true, y_trained_pred)
            latest_metric = get_regression_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.r2_score - latest_metric.r2_score
            
            if self.model_evaluation_config.change_threshold < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                trained_model_path=train_model_file_path,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric)
            model_eval_report = model_evaluation_artifact._asdict
            # save report
            write_yaml_file(file_path=self.model_evaluation_config.report_file_path, content=model_eval_report)
            logger.info(f'Model evaluation artifact: {model_evaluation_artifact}')
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e,sys)