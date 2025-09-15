import gradio as gr
from transformers import pipeline
import torch

# تحميل الموديل
MODEL_PATH = "./multilang_model"
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    device=device
)

# تحويل LABEL_X → positive/negative
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    "LABEL_2": "negative",
    "LABEL_3": "positive"
}

def predict_review(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    mapped_label = label_map.get(label, "unknown")
    return f"Predicted sentiment: {mapped_label}, confidence: {score:.4f}"

# واجهة Gradio
interface = gr.Interface(
    fn=predict_review,
    inputs=gr.Textbox(lines=5, placeholder="اكتب مراجعة بالعربي أو الإنجليزي..."),
    outputs="text",
    title="Multilingual Review Classifier",
    description="ادخل مراجعة بالإنجليزي أو العربي واحصل على التوقع."
)

interface.launch()
