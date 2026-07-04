import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class VehicleClassifierCNN(nn.Module):
    def __init__(self):
        super(VehicleClassifierCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

st.set_page_config(page_title="Klasifikasi Kendaraan")
st.title("Sistem Klasifikasi Kendaraan Berbasis Citra")
st.write("Upload gambar kendaraan **(Bus, Mobil, atau Motor)**.")

@st.cache_resource
def load_model():
    model = VehicleClassifierCNN()
    # Load bobot dan pasangkan ke CPU
    model.load_state_dict(torch.load('model_best.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Bus', 'Mobil', 'Motor'] 

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    st.write("Prediksi")
    
    input_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    confidence, predicted_idx = torch.max(probabilities, 0)
    predicted_class = class_names[predicted_idx.item()]
    
    st.success(f"Hasil Prediksi: **{predicted_class}**")

    for i, class_name in enumerate(class_names):
        prob_score = probabilities[i].item() * 100

        st.write(f"{class_name}: {prob_score:.2f}%")
        
        st.progress(int(prob_score))