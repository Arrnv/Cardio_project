import sys
import os

import certifi
ca = certifi.where()
import numpy as np
from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from major.exception.exception import MajorException
from major.logging.logger import logging
from major.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd
from pydantic import BaseModel


from major.utils.main_utils.utils import load_object
from major.utils.ml_utils.model.estimator import MajorModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from major.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from major.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from major.components.data_transformation import DataTransformation
class CustomData(BaseModel):
    age: int
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    cardio:int
    age_years: int
    bmi:float
    bp_category:str
    bp_category_encoded:str

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from fastapi.templating import Jinja2Templates
# templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

    
@app.post("/predict")
async def predict_route(data: CustomData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Apply data transformations
        transformed_df = DataTransformation.transform_data(input_df)
        
        # Load the preprocessor object
        preprocessor = load_object("final_model/preprocessor.pkl")
        
        # Preprocess the data
        processed_data = preprocessor.transform(transformed_df)
        
        # Load the trained model (update this path to your actual model path)
        model = load_object("final_model/model.pkl")
        
        # Perform prediction
        predictions = model.predict(processed_data)
        
        # Return the predictions as a response
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise MajorException(e, sys)


    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)
