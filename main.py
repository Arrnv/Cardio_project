from major.components.Data_ingestion import DataIngestion
from major.components.data_validation import DataValidation

from major.exception.exception import MajorException
from major.logging.logger import logging
from major.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from major.entity.config_entity import TrainingPipelineConfig 
from major.components.data_transformation import DataTransformation
import sys

if __name__ == "__main__":
    try:
        logging.info("initate data ingestion")
        trainingpipelineconfig = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config)
        dataingestionartifacts = data_ingestion.InitateDataIngestion()
        print(dataingestionartifacts)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        logging.info("data validation statred")
        data_validation = DataValidation(data_ingestion_config,data_validation_config)
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.InitiateDataTransformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")
    except Exception as e:
        raise MajorException(e ,sys)