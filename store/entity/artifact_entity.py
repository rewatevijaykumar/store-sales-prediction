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