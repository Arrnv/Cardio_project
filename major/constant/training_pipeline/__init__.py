import os
import sys
import numpy as np # type: ignore
import pandas as pd# type: ignore


SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
"""
Data ingestion related constant starts with DATA_INGESTION VAR NAME
"""

DATA_INGESTION_COLLECTION_NAME: str = "cardio_data"
DATA_INGESTION_DATABASE_NAME: str = "MAJOR_AI"
DATA_INGESTION_DIR_NAME: str = "data_ingested"
DATA_INGESTION_FEATURE_STORE_DIR: str = "featurea_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
defining common const variable for training pipeline
"""
TRAGET_COLLUMN = "cardio_disease"
PIPELINE_NAME:str = "Major"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "cardio_data_processed.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

"""
data validation realated constant
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validation"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRAFT_REPORT_DIR: str = "draft_report"
DATA_VALIDATION_DRAFT_REPORT_FILE_NAME:str = "report.yaml"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"
