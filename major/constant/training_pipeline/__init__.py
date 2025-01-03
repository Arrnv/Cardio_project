import os
import sys
import numpy as np # type: ignore
import pandas as pd# type: ignore

"""
Data ingestion related constant starts with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "cardio_data"
DATA_INGESTION_DATABASE_NAME: str = "MAJOR_AI"
DATA_INGESTION_DIR_NAME: str = "data_ingested"
DATA_INGESTION_FEATURE_STORE_DIR: str = "featurea_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.25

"""
defining common const variable for training pipeline
"""
TRAGET_COLLUMN = "cardio"
PIPELINE_NAME:str = "Major"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "cardio_data_processed.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
