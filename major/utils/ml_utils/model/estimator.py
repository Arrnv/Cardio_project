from major.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

import os
import sys
import numpy as np

from major.exception.exception import MajorException
from major.logging.logger import logging

class MajorModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise MajorException(e,sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise MajorException(e,sys)
        
    def predict1(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            x_transform = np.c_[x_transform, np.array(x_transform)]
            y_hat = self.model.predict(x_transform[:,:-1])
            return y_hat
        except Exception as e:
            raise MajorException(e,sys)