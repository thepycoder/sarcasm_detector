import os
from pathlib import Path
import time

import joblib
import pandas as pd
from clearml import Dataset, Task
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from uuid import uuid4

from utils import plot_confusion_matrix


class SklearnTrainer():
    def __init__(self, model='LinearRegression', seed=42, subset_size=0):
        self.task = Task.init(project_name="sarcasm_detector", task_name="Sklearn Training")
        self.task.set_parameter("model", model)
        self.task.set_parameter("Seed", seed)
        self.task.set_parameter("Subset Size", subset_size)

        self.seed = seed
        self.model = model
        self.subset = subset_size
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
            dataset_project="sarcasm_detector",
            dataset_name="sarcasm_dataset",
            alias="sarcasm_dataset"
        ).get_local_copy())

        dataset = load_dataset(
            "csv",
            data_files=[str(local_dataset_path / csv_path) for csv_path in os.listdir(local_dataset_path)],
            split="all"
        )
        dataset = dataset.train_test_split(
            test_size=0.2,
            shuffle=True,
            seed=self.seed
        )
        if self.subset:
            dataset['train'] = dataset['train'].select(range(self.subset))
        dataset = dataset.filter(lambda x: bool(x['comment']))

        return dataset['train']['comment'], dataset['train']['label'], dataset['test']['comment'], dataset['test']['label']
    

    def train(self):
        train, y_train, test, y_test = self.get_data()

        start_training = time.time()
        self.pipeline.fit(train, y_train)
        self.task.get_logger().report_single_value("train_runtime", time.time() - start_training)

        y_pred = self.pipeline.predict(test)
        self.task.get_logger().report_single_value("Accuracy", accuracy_score(y_test, y_pred))
        plot_confusion_matrix(
            y_test,
            y_pred,
            ["NORMAL", "SARCASTIC"],
            figsize=(8, 8),
            title=f"{self.model} Confusion Matrix"
        )

        joblib.dump(self.pipeline, f"my_awesome_model/sklearn_classifier_{uuid4()}.joblib")


if __name__ == '__main__':
    sarcasm_trainer = SklearnTrainer(subset_size=1000)
    sarcasm_trainer.train()