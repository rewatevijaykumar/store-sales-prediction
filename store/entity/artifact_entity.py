from collections import namedtuple

DataIngestionArtifact = namedtuple('DataIngestionArtifact', 
                                   ['trained_file_path', 
                                    'test_file_path'])

DataValidationArtifact = namedtuple('DataValidationArtifact', 
                                    ['validation_status',
                                     'valid_train_file_path',
                                     'valid_test_file_path',
                                     'invalid_train_file_path',
                                     'invalid_test_file_path',
                                     'drift_report_file_path'])

DataTransformationArtifact = namedtuple('DataTransformationArtifact',
                                        ['transformed_object_file_path',
                                         'transformed_train_file_path',
                                         'transformed_test_file_path',
                                         'transformed_target_object_file_path'])

RegressionMetricArtifact = namedtuple('RegressionMetricArtifact',[ 
                                            'mse',
                                            'mae',
                                            'r2_score'])

ModelTrainerArtifact = namedtuple('ModelTrainerArtifact',
                                  ['trained_model_file_path',
                                   'train_metric_artifact',
                                   'test_metric_artifact'])

ModelEvaluationArtifact = namedtuple('ModelEvaluationArtifact',
                                     ['is_model_accepted',
                                      'improved_accuracy',
                                      'best_model_path',
                                      'trained_model_path',
                                      'train_model_metric_artifact',
                                      'best_model_metric_artifact'])

ModelPusherArtifact = namedtuple('ModelPusherArtifact',
                                 ['saved_model_path',
                                  'model_file_path'])
