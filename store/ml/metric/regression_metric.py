from store.entity.artifact_entity import RegressionMetricArtifact
from store.exception import CustomException
from store.logger import logger
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import os,sys

def get_regression_score(y_true, y_pred)->RegressionMetricArtifact:
    try:
        model_mse = mean_squared_error(y_true, y_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        model_r2_score = r2_score(y_true, y_pred)
        regression_metric = RegressionMetricArtifact(
            mse=model_mse,
            mae=model_mae,
            r2_score=model_r2_score,
        )
        return regression_metric
    except Exception as e:
        raise CustomException(e,sys)
    