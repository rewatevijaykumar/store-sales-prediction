from collections import namedtuple

DataIngestionArtifact = namedtuple(
    'DataIngestionArtifact', 
    ['trained_file_path', 
     'test_file_path'])