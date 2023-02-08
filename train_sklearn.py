import os
from pathlib import Path
from clearml import Dataset, Task
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from utils import plot_confusion_matrix


class SklearnTrainer():
    def __init__(self, model='LinearRegression'):
        self.task = Task.init(project_name="sarcasm-detector", task_name="Sklearn Training")
        self.task.set_parameter("model", model)
        self.model = model
        self.pipeline = self.create_pipeline()
    

    def create_pipeline(self):
        # Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)

        # Model
        model = None
        if self.model == "LinearRegression":
            # multinomial logistic regression a.k.a softmax classifier
            cfg = {
                "C": 1,
                "n_jobs": 4,
                "solver": 'lbfgs',
                "random_state": 17,
                "verbose": 1
            }
            self.task.connect(cfg)
            model = LogisticRegression(
                **cfg
            )
        
        # Pipeline
        return Pipeline([('vectorizer', vectorizer), ('model', model)])

    
    def get_data(self):
        local_dataset_path = Path(Dataset.get(
            dataset_project="sarcasm-detector",
            dataset_name="reddit_kaggle",
            alias="reddit_kaggle"
        ).get_local_copy())

        train_df = pd.read_csv(str(local_dataset_path / "train-balanced-sarcasm.train.csv")).dropna()
        test_df = pd.read_csv(str(local_dataset_path / "train-balanced-sarcasm.test.csv")).dropna()

        train, y_train = train_df["comment"], train_df["label"]
        test, y_test = test_df["comment"], test_df["label"]

        return train, y_train, test, y_test
    

    def train(self):
        train, y_train, test, y_test = self.get_data()
        self.pipeline.fit(train, y_train)

        y_pred = self.pipeline.predict(test)
        self.task.report_single_value("Accuracy Score", accuracy_score(y_test, y_pred))
        plot_confusion_matrix(y_test, y_pred, self.pipeline.named_steps['model'].classes_, figsize=(8, 8))


if __name__ == '__main__':
    sarcasm_trainer = SklearnTrainer()
    sarcasm_trainer.train()