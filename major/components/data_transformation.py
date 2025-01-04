import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from major.logging.logger import logging
from major.constant.training_pipeline import TRAGET_COLLUMN
from major.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from major.entity.config_entity import DataTransformationConfig
from major.exception.exception import MajorException
from major.logging.logger import logging
from major.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig
                 ):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
            
        except Exception as e:
            raise MajorException(e,sys)
        
    @staticmethod    
    def read_data(file_path)-> pd.DataFrame:
        try:
           return pd.read_csv(file_path)
            
        except Exception as e:
            raise MajorException(e,sys)
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        """
            It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
            and returns a Pipeline object with the KNNImputer object as the first step.

            Args:
            cls: DataTransformation

            Returns:
            A Pipeline object
        """
        try:   
            logging.info(
                "Entered get_data_trnasformer_object method of Trnasformation class"
            )
            
            numeric_features = [
                'age', 'height', 'systolic_b_pressure', 'diastolic_b_pressure', 'glucose', 
                'bmi', 'cholesterol_bmi_interaction', 'active_with_disease'
            ]

            categorical_features = [
                'blood_pressure_category_prehypertension', 
                'blood_pressure_category_hypertension', 
                'age_group_middle_aged', 
                'age_group_elderly'
            ]
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),  
                    ('cat', OneHotEncoder(drop='first'), categorical_features) 
                ]
            )
            return preprocessor
        except Exception as e:
            raise MajorException(e,sys)
        
    # Function to apply transformations
    def transform_data(df):
                df['age'] = df['age'] / 365.25
                df = df.rename(columns={
                    'ap_hi': 'systolic_b_pressure',
                    'ap_lo': 'diastolic_b_pressure',
                    'gluc': 'glucose',
                    'alco': 'alcohol',
                    'active': 'physically_active',
                    'cardio': 'cardio_disease'
                })

                # Data cleaning
                df = df[(df['height'] >= 100) & (df['height'] <= 200)]
                df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
                df = df[(df['systolic_b_pressure'] >= 90) & (df['systolic_b_pressure'] <= 250)]
                df = df[(df['diastolic_b_pressure'] >= 60) & (df['diastolic_b_pressure'] <= 150)]
                df = df[df['cholesterol'].isin([1, 2, 3])]
                df = df[df['glucose'].isin([1, 2, 3])]
                df = df.dropna()
                df = df[df['gender'].isin([1, 2])]

                # Feature engineering
                df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
                categories = ['normal', 'prehypertension', 'hypertension']
                df['blood_pressure_category'] = pd.cut(
                    df['systolic_b_pressure'], bins=[0, 120, 140, float('inf')], labels=categories
                )
                bins = [30, 45, 60, 80]
                labels = ['young', 'middle_aged', 'elderly']
                df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
                df['smoke_and_alcohol'] = ((df['smoke'] == 1) & (df['alcohol'] == 1)).astype(int)
                df['pressure_ratio'] = df['systolic_b_pressure'] / df['diastolic_b_pressure']
                df['cholesterol_bmi_interaction'] = df['cholesterol'] * df['bmi']
                df['height_weight_ratio'] = df['height'] / df['weight']
                df['active_with_disease'] = ((df['physically_active'] == 1) & (df['cardio_disease'] == 1)).astype(int)

                non_numeric_columns = df.select_dtypes(exclude=['number']).columns
                df_encoded = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)

                features_to_drop = [
                    'gender', 'smoke', 'alcohol', 'physically_active',
                    'pressure_ratio', 'weight', 'height_weight_ratio', 'cholesterol'
                ]
                features_to_drop = [col for col in features_to_drop if col in df_encoded.columns]
                df_encoded_cleaned = df_encoded.drop(columns=features_to_drop)

                return df_encoded_cleaned
            
    def InitiateDataTransformation(self) -> DataTransformationArtifact:
        logging.info("Entered data transformation")
        try:
            logging.info("Starting data transformation")

            # Read training and testing data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_training_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_testing_file_path)

            # Apply transformations
            train_df = self.transform_data(train_df)
            test_df = self.transform_data(test_df)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TRAGET_COLLUMN], axis=1)
            target_feature_train_df = train_df[TRAGET_COLLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TRAGET_COLLUMN], axis=1)
            target_feature_test_df = test_df[TRAGET_COLLUMN].replace(-1, 0)

            # Preprocessing
            preprocess = self.get_data_transformer_object()
            preprocess_obj = preprocess.fit(input_feature_train_df)
            transformed_input_train_feature = preprocess_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocess_obj.transform(input_feature_test_df)

            # Combine features and targets
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save artifacts
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocess_obj)
            save_object("final_model/preprocessor.pkl", preprocess_obj)

            # Preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise Exception(f"Error in Data Transformation: {e}")
