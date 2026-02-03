import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pickle
import gradio as gr

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

num_classes = len(class_names)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_base_models():
    models_dict = {}

    m1 = models.resnet50(pretrained=False)
    m1.fc = nn.Linear(2048, num_classes)
    m1.load_state_dict(torch.load("resnet50.pth", map_location="cpu"))
    m1.eval()
    models_dict["resnet50"] = m1

    m2 = models.efficientnet_b0(pretrained=False)
    m2.classifier[1] = nn.Linear(1280, num_classes)
    m2.load_state_dict(torch.load("efficientnet_b0.pth", map_location="cpu"))
    m2.eval()
    models_dict["efficientnet_b0"] = m2

    m3 = models.vit_b_16(pretrained=False)
    m3.heads.head = nn.Linear(768, num_classes)
    m3.load_state_dict(torch.load("vit_b_16.pth", map_location="cpu"))
    m3.eval()
    models_dict["vit_b_16"] = m3

    m4 = models.mobilenet_v2(pretrained=False)
    m4.classifier[1] = nn.Linear(1280, num_classes)
    m4.load_state_dict(torch.load("mobilenet_v2.pth", map_location="cpu"))
    m4.eval()
    models_dict["mobilenet_v2"] = m4

    return models_dict


base_models = load_base_models()

with open("meta_classifier.pkl", "rb") as f:
    meta_clf = pickle.load(f)

def predict(image):
    img = transform(image).unsqueeze(0)

    preds = []
    for name, model in base_models.items():
        with torch.no_grad():
            logits = model(img)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
        preds.extend(probs)

    preds = np.array(preds).reshape(1, -1)
    final_pred = meta_clf.predict(preds)[0]
    final_prob = max(meta_clf.predict_proba(preds)[0])

    return f"Prediction: {class_names[final_pred]} (Confidence: {final_prob:.2f})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor Stacking Ensemble Classifier",
    description="Upload an image for prediction."
)

interface.launch()
