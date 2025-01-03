import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from major.logging.logger import logging
from major.constant.training_pipeline import TRAGET_COLLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from major.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from major.entity.config_entity import DataTransformationConfig
from major.exception.exception import MajorException
from major.logging.logger import logging
from major.utils.main_utils.utils import save_numpy_array_data, save_object