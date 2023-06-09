from collections import namedtuple

TrainingPipelineConfig = namedtuple('PipelineConfig',['pipeline_name','artifact_dir'])

DataIngestionConfig = namedtuple('DataIngestionConfig',['data_ingestion_dir',
                                                          'feature_store_file_path',
                                                          'training_file_path',
                                                          'testing_file_path',
                                                          'train_test_split_ratio',
                                                          'collection_name'])

DataValidationConfig = namedtuple('DataValidationConfig', ['data_validation_dir', 
                                                           'valid_data_dir',
                                                           'invalid_data_dir',
                                                           'valid_train_file_path',
                                                           'valid_test_file_path',
                                                           'invalid_train_file_path',
                                                           'invalid_test_file_path',
                                                           'drift_report_file_path'])

DataTransformationConfig = namedtuple('DataTransformationConfig',['data_transformation_dir', 
                                                                  'transformed_train_file_path',
                                                                  'transformed_test_file_path',
                                                                  'transformed_object_file_path',
                                                                  'transformed_target_object_file_path'])


ModelTrainerConfig = namedtuple('ModelTrainerConfig',
                                ['model_trainer_dir',
                                 'trained_model_file_path',
                                 'expected_accuracy',
                                 'overfitting_underfitting_threshold'])

ModelEvaluationConfig = namedtuple('ModelEvaluationConfig',
                                ['model_evaluation_dir',
                                 'change_threshold',
                                 'report_file_path'])

ModelPusherConfig = namedtuple('ModelPusherConfig',
                                ['model_pusher_dir',
                                 'model_file_path',
                                 'saved_model_path',
                                 'target_object_file_path'])