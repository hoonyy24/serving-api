# app.py
import io
import torch
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify
from model import Classifier  


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(num_classes=14)             
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()  

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),       
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file in request"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    input_tensor = preprocess(image).unsqueeze(0).to(device) 

    
    with torch.no_grad():
        logits = model(input_tensor)  
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    confidence = probs[pred_idx]

    return jsonify({
        "predicted_class": pred_idx,
        "confidence": confidence,
        "all_probabilities": probs
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
