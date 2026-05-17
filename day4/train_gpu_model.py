# train_gpu_model.py
import torch
import torchvision.models as models
import os

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)
model.eval()
# Save weights
torch.save(model.state_dict(), 'resnet18.pth')
print(f"Model saved: {os.path.getsize('resnet18.pth') / 1024 / 1024:.1f} MB")
# Output: Model saved: 44.7 MB