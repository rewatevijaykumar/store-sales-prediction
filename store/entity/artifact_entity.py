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