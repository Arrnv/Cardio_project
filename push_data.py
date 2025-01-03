# from pymongo import MongoClient
# from dotenv import load_dotenv
# load_dotenv()
# import os 


# MONGO_URL = os.getenv("MONGO_URL")
# try:
#     client = MongoClient(MONGO_URL)
#     print("Connection successful!")
# except Exception as e:
#     print(f"Connection failed: {e}")

import os 
import sys
import json
from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

import certifi
ca = certifi.where()

import numpy as np
import pandas as pd
# import pymongo
import pymongo 
from major.exception.exception import MajorException
from major.logging.logger import logging

class Major_data_extract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise MajorException(e, sys)
    
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise MajorException(e,sys)
        
    def inser_data_mongodb(self, records, database, collection):
        try:
            self.client = pymongo.MongoClient(MONGO_URL, tlsCAFile=ca)
            self.database = self.client[database]
            self.collection = self.database[collection]
            self.collection.insert_many(records)
            return len(records)
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            raise MajorException(e, sys)


        
if __name__ == "__main__":
    FILE_PATH = "data/cardio_data_processed.csv"
    DATA_BASE = "MAJOR_AI"
    COLLECTION = "cardio_data"
    
    obj  = Major_data_extract()
    rec = obj.csv_to_json_converter(file_path=FILE_PATH)
    print(rec)
    no_of_record = obj.inser_data_mongodb(rec,DATA_BASE,COLLECTION)
    print(no_of_record)