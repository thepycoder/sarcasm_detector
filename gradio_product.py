import gradio as gr
import joblib
import time
from transformers import pipeline

# transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-56828", device='cuda:0')
transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-56828", device='cpu')
sklearn_pipeline = joblib.load("my_awesome_model/sklearn_classifier_30c256d0-b43e-4d03-92ae-5c45d96dddcb.joblib")

input_text = gr.Textbox(lines=2, placeholder="Sentence Here...")

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


transformers_callback = gr.CSVLogger()
logistic_callback = gr.CSVLogger()

demo = gr.Blocks()
with demo:
    with gr.Row():
        gr.HTML("<p style='text-align:center;'><img src='https://clear.ml/wp-content/uploads/2020/12/clearml-logo.svg' style='display:inline-block; margin:auto;' width='30%' /></p>")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox()
            b1 = gr.Button("Classify Sarcasm")
    with gr.Row():
        with gr.Column():
            output_transformer = gr.Textbox(label="Model 1")
            b2 = gr.Button("Model 1 was Wrong")
        with gr.Column():
            output_logistic = gr.Textbox(label="Model 2")
            b3 = gr.Button("Model 2 was Wrong")

    # This needs to be called at some point prior to the first call to callback.flag()
    transformers_callback.setup([text, output_transformer], "flagged_transformer")
    logistic_callback.setup([text, output_logistic], "flagged_logistic")

    b1.click(classify_transformer, inputs=text, outputs=output_transformer)
    b1.click(classify_logistic, inputs=text, outputs=output_logistic)

    b2.click(lambda *args: transformers_callback.flag(args), [text, output_transformer], None, preprocess=False)
    b3.click(lambda *args: logistic_callback.flag(args), [text, output_logistic], None, preprocess=False)

demo.launch()