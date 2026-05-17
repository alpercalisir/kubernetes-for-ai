# app_gpu.py
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)
# Falls back to CPU on Mac - identical API, slower inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('resnet18.pth', map_location=device))
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = Image.open(
        io.BytesIO(request.files['image'].read())
    ).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_id = torch.topk(probs, 5)
    return jsonify({
        'predictions': [
            {
                'class_id': top5_id[i].item(),
                'probability': top5_prob[i].item()
            }
            for i in range(5)
        ],
        'device': str(device)
    })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))