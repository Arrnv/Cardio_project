from major.components.Data_ingestion import DataIngestion
from major.exception.exception import MajorException
from major.logging.logger import logging
from major.entity.config_entity import DataIngestionConfig
from major.entity.config_entity import TrainingPipelineConfig 
import sys

if __name__ == "__main__":
    try:
        logging.info("initate data ingestion")
        trainingpipelineconfig = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config)
        dataingestionartifacts = data_ingestion.InitateDataIngestion()
        print(dataingestionartifacts)
        
    except Exception as e:
        raise MajorException(e ,sys)