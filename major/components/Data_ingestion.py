from major.exception.exception import MajorException
from major.logging.logger import logging

# configration from the data ingestion config
from major.entity.config_entity import DataIngestionConfig
from major.entity.artifact_entity import DataIngestionArtifact
import os 
import sys
import pymongo
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MajorException(e, sys)
        
    def export_json_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_URL)
            collection=self.mongo_client[database_name][collection_name]
            df =  pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df  = df.drop(columns=["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise MajorException(e, sys)
        
    def export_data_to_feature_store(self, dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False, header=True)
            return dataframe 
        except Exception as e:
            raise MajorException(e, sys)
        
    def split_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("performing the train test split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Exporting train test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False, header=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False, header=False)
            logging.info("Exporting the train test files")

            
        except Exception as e:
            raise MajorException(e, sys)
        
    def InitateDataIngestion(self):
        try:
            dataframe = self.export_json_as_dataframe()
            dataframe =  self.export_data_to_feature_store(dataframe)
            self.split_data_into_feature_store(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return dataingestionartifact
        except Exception as e:
            raise MajorException(e,sys)