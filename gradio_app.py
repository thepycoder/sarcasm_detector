import gradio as gr
import joblib
from transformers import pipeline

transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-best", device='cuda:0')
sklearn_pipeline = joblib.load("my_awesome_model/sklearn_classifier_ffa779cd-0944-4d7b-a5da-cb5f749f06ee.joblib")

input_text = gr.Textbox(lines=2, placeholder="Sentence Here...")

def classify_transformer(sentence):
    sarcastic = transformer_pipeline(sentence)[0]
    return f"{sarcastic['label']}: {sarcastic['score']}"

def classify_sklearn(sentence):
    sarcastic = sklearn_pipeline.predict_proba([sentence])[0]
    if sarcastic[0] > sarcastic[1]:
        label = "NORMAL"
        score = sarcastic[0]
    else:
        label = "SARCASTIC"
        score = sarcastic[1]
    return f"{label}: {score}"


transformer = gr.Interface(
    fn=classify_transformer,
    inputs=input_text,
    outputs=gr.Textbox(label="DistilBERT Output"),
)

sklearn = gr.Interface(
    fn=classify_sklearn,
    inputs=input_text,
    outputs=gr.Textbox(label="LogisticRegression Ouput"),
)

demo = gr.Parallel(transformer, sklearn)
demo.launch()
