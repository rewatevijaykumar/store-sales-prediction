from collections import namedtuple

TrainingPipelineConfig = namedtuple('PipelineConfig',['pipeline_name','artifact_dir'])

DataIngestionConfig = namedtuple('DataIngestionConfig',['data_ingestion_dir',
                                                          'feature_store_file_path',
                                                          'training_file_path',
                                                          'testing_file_path',
                                                          'train_test_split_ratio',
                                                          'collection_name'])