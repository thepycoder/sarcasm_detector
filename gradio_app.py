import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model="./my_awesome_model/checkpoint-56828", device='cuda:0')

def classify(sentence):
    sarcastic = classifier(sentence)
    return sarcastic

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=2, placeholder="Sentence Here..."),
    outputs="text",
)
demo.launch()
