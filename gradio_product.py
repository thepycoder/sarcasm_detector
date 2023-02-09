import csv
import os
from pathlib import Path
import re
from typing import Any, List
from uuid import uuid4
from gradio.flagging import FlaggingCallback
from gradio.components import IOComponent
import gradio as gr
import joblib
import time
from transformers import pipeline

# transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-56828", device='cuda:0')
transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-56828", device='cpu')
sklearn_pipeline = joblib.load("my_awesome_model/sklearn_classifier_30c256d0-b43e-4d03-92ae-5c45d96dddcb.joblib")

demo = gr.Blocks()
with demo:

    def classify_transformer(sentence):
        start = time.time()
        sarcastic = transformer_pipeline(sentence)[0]
        time_taken = time.time() - start
        return f"LABEL: {sarcastic['label']}\nCERTAINTY: {sarcastic['score']:.2f}\nCOMPUTE TIME: {time_taken:.5f}"

    def classify_logistic(sentence):
        start = time.time()
        sarcastic = sklearn_pipeline.predict_proba([sentence])[0]
        time_taken = time.time() - start
        if sarcastic[0] > sarcastic[1]:
            label = "NORMAL"
            score = sarcastic[0]
        else:
            label = "SARCASTIC"
            score = sarcastic[1]
        return f"LABEL: {label}\nCERTAINTY: {score:.2f}\nCOMPUTE TIME: {time_taken:.5f}"

    def parse_output_to_label(output):
        if not output.startswith("LABEL"):
            return [output]
        # Extract the label and certainty value from the text
        result = re.search(r"LABEL: (\w+)\nCERTAINTY: ([\d\.]+)", output)

        # Check if the label is "NORMAL" or "SARCASTIC"
        if result:
            label = result.group(1)
            certainty = float(result.group(2))
            if label == "NORMAL":
                return [0, certainty]
            elif label == "SARCASTIC":
                return [1, certainty]


    class ClearMLDatasetLogger(FlaggingCallback):
        def __init__(self):
            pass
        
        def setup(self, components: List[IOComponent], file_prefix: str):
            self.components = components
            self.flagging_dir = "flagged"
            self.file_prefix = file_prefix
            os.makedirs(self.flagging_dir, exist_ok=True)
        
        def flag(
            self,
            flag_data: List[Any],
            amount_labeled_var: int,
            csv_filename: str
        ) -> int:
            flagging_dir = self.flagging_dir
            log_filepath = Path(flagging_dir) / (self.file_prefix + str(csv_filename) + ".csv")
            print(amount_labeled_var.value)
            csv_data = []
            for component, sample in zip(self.components, flag_data):
                print(component.__dict__)
                if type(component) == gr.components.Textbox:
                    csv_data += \
                        parse_output_to_label(component.deserialize(
                            sample
                        ))
            with open(log_filepath, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_data)
            
            new_count = amount_labeled_var.value + 1
            return new_count, f"{new_count} labeled samples"


    transformers_callback = ClearMLDatasetLogger()
    logistic_callback = ClearMLDatasetLogger()
    amount_labeled_var = gr.State(0)
    csv_filename = gr.State(f"{uuid4()}.csv")
    with gr.Row():
        gr.HTML("<p style='text-align:center;'><img src='https://clear.ml/wp-content/uploads/2020/12/clearml-logo.svg' style='display:inline-block; margin:auto;' width='30%' /></p>")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Model input sentence")
            b1 = gr.Button("Classify Sarcasm")
    with gr.Row():
        with gr.Column():
            output_transformer = gr.Textbox(label="Model 1")
            b2 = gr.Button("Model 1 was Wrong")
        with gr.Column():
            output_logistic = gr.Textbox(label="Model 2")
            b3 = gr.Button("Model 2 was Wrong")
    with gr.Row():
        with gr.Column():
            counter = gr.Label(f"{amount_labeled_var.value} labeled samples")
        with gr.Column():
            b4 = gr.Button(f"Package Labeled Samples")

    # This needs to be called at some point prior to the first call to callback.flag()
    transformers_callback.setup([text, output_transformer, csv_filename, amount_labeled_var], "flagged_transformer")
    logistic_callback.setup([text, output_logistic, csv_filename, amount_labeled_var], "flagged_logistic")

    # Run the models
    b1.click(classify_transformer, inputs=text, outputs=output_transformer)
    b1.click(classify_logistic, inputs=text, outputs=output_logistic)

    # Serve as a labeling tool
    b2.click(lambda *args: transformers_callback.flag(args, amount_labeled_var, csv_filename.value), [text, output_transformer, csv_filename], [amount_labeled_var, counter], preprocess=False)
    b3.click(lambda *args: logistic_callback.flag(args, amount_labeled_var, csv_filename.value), [text, output_logistic, csv_filename], [amount_labeled_var, counter], preprocess=False)

    # Package the current labels and ship them as a ClearML Dataset
    # b4.click()

demo.launch(debug=True)