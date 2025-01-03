from major.components.Data_ingestion import DataIngestion
from major.components.data_validation import DataValidation

from major.exception.exception import MajorException
from major.logging.logger import logging
from major.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from major.entity.config_entity import TrainingPipelineConfig 
from major.components.data_transformation import DataTransformation
from major.components.model_trainer import ModelTrainer, ModelTrainerConfig
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.InitateDataIngestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.InitiateDataTransformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")

        logging.info("Model Training stared")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")
    except Exception as e:
        raise MajorException(e ,sys)