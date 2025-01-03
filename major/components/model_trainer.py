import os
import sys
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import mlflow
from urllib.parse import urlparse

from major.exception.exception import MajorException
from major.logging.logger import logging

from major.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from major.entity.config_entity import ModelTrainerConfig

from major.utils.ml_utils.model.estimator import MajorModel
from major.utils.main_utils.utils import load_object, save_object, load_numpy_array_data, evaluate_models
from major.utils.ml_utils.metric.classification_metric import get_classification_score

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifact
        except Exception as e:
            raise MajorException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training initiated.")

            models = {
                "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
                "SVC": SVC(probability=True, random_state=42),
                "SGDClassifier": SGDClassifier(loss="log_loss", random_state=42),
                "VotingClassifier": VotingClassifier(estimators=[
                    ("hgb", HistGradientBoostingClassifier(random_state=42)),
                    ("svc", SVC(probability=True, random_state=42)),
                    ("sgd", SGDClassifier(loss="log_loss", random_state=42))
                ], voting="soft")
            }

            params = {
                "HistGradientBoosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_iter": [100, 200],
                },
                "SVC": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                },
                "SGDClassifier": {
                    "alpha": [0.0001, 0.001, 0.01],
                    "penalty": ["l2", "elasticnet"],
                },
                "VotingClassifier": {}
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Select the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Train final model
            best_model.fit(X_train, y_train)
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate classification metrics
            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track experiments (optional MLflow integration)
            self.track_mlflow(best_model, train_metrics)
            self.track_mlflow(best_model, test_metrics)

            # Save the best model
            preprocessor = load_object(self.data_transformation_artifacts.transformed_object_file_path)
            final_model = MajorModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=final_model)

            logging.info(f"Best model: {best_model_name} with accuracy: {model_report[best_model_name]}")
            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )

        except Exception as e:
            raise MajorException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training and testing data.")
            train_file_path = self.data_transformation_artifacts.transformed_train_file_path
            test_file_path = self.data_transformation_artifacts.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise MajorException(e, sys)

    def track_mlflow(self,best_model,classificationmetric):
        mlflow.set_registry_uri("https://dagshub.com/Arrnv/Cardio_project.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")
