from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_path:str
    testing_file_path:str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_training_file_path: str
    valid_testing_file_path: str
    invalid_training_file_path: str
    invalid_testing_file_path: str
    drift_report_file_path: str
    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str