import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    # Tải mô hình ResNet-50 đã được huấn luyện trên ImageNet
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Loại bỏ lớp fully connected cuối cùng
    # Vector output sẽ là feature từ lớp average pooling ngay trước đó
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    # Chuyển mô hình sang chế độ đánh giá (không training) và đưa lên thiết bị
    feature_extractor.eval()
    feature_extractor.to(DEVICE)
    
    return feature_extractor

def get_image_transform():
    """
    Tạo pipeline để tiền xử lý ảnh đầu vào cho mô hình CNN.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_features(image_path, model, transform):
    """
    Trích xuất feature vector từ một file ảnh.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(DEVICE)
        
        with torch.no_grad():
            features = model(batch_t)
        
        # Làm phẳng vector và chuyển về dạng numpy
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"⚠️ Lỗi khi xử lý ảnh {image_path}: {e}")
        return None


