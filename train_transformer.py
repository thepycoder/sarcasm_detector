import os
from pathlib import Path

import evaluate
import numpy as np
from clearml import Dataset, Task
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from utils import plot_confusion_matrix


def cast_keys_to_string(d, changed_keys=dict()):
    nd = dict()
    for key in d.keys():
        if not isinstance(key, str):
            casted_key = str(key)
            changed_keys[casted_key] = key
        else:
            casted_key = key
        if isinstance(d[key], dict):
            nd[casted_key], changed_keys = cast_keys_to_string(d[key], changed_keys)
        else:
            nd[casted_key] = d[key]
    return nd, changed_keys

def cast_keys_back(d, changed_keys):
    nd = dict()
    for key in d.keys():
        if key in changed_keys:
            original_key = changed_keys[key]
        else:
            original_key = key
        if isinstance(d[key], dict):
            nd[original_key], changed_keys = cast_keys_back(d[key], changed_keys)
        else:
            nd[original_key] = d[key]
    return nd, changed_keys


class SarcasmTrainer:
    def __init__(self, seed=42, subset_size=0):
        self.accuracy = evaluate.load("accuracy")
        self.classes = ["NORMAL", "SARCASTIC"]
        self.id2label = {0: "NORMAL", 1: "SARCASTIC"}
        self.label2id = {"NORMAL": 0, "SARCASTIC": 1}
        Task.current_task().set_parameter("Seed", seed)
        self.seed = seed
        Task.current_task().set_parameter("Subset Size", subset_size)
        self.subset = subset_size

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        self.model.to('cuda')

        self.trainer = None

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

        return dataset

    def tokenize_data(self, dataset):
        preprocess_function = lambda examples: self.tokenizer(examples["comment"], truncation=True)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        return data_collator, tokenized_dataset

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        plot_confusion_matrix(labels, predictions, self.classes, title='DistilBERT Confusion Matrix')
        accuracy = self.accuracy.compute(predictions=predictions, references=labels)
        Task.current_task().get_logger().report_single_value("Accuracy", accuracy['accuracy'])
        return accuracy

    def train(self):
        dataset = self.get_data()
        data_collator, tokenized_dataset = self.tokenize_data(dataset)

        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            dataloader_num_workers=0,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        # Allow ClearML access to the training args and allow it to override the arguments for remote execution
        args_class = type(training_args)
        args, changed_keys = cast_keys_to_string(training_args.to_dict())
        Task.current_task().connect(args)
        training_args = args_class(**cast_keys_back(args, changed_keys)[0])

        print(f"Training with params: {training_args}")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()


if __name__ == '__main__':
    Task.add_requirements("torch")
    Task.init(project_name="sarcasm_detector", task_name="DistilBert Training")
    sarcasm_trainer = SarcasmTrainer(subset_size=1000)
    sarcasm_trainer.train()
