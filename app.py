import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# -----------------------------
# 1. Define VGG6 model (3-channel input)
# -----------------------------
class VGG6(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG6, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3 channels (RGB)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),  # for 48x48 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# 2. Load pretrained weights
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG6(num_classes=7).to(device)

state_dict = torch.load("models/modelvgg6_like.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# 3. Preprocessing for RGB
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels
])

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # keep RGB for model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # only for detection

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = rgb_frame[y:y+h, x:x+w]

        # Preprocess ROI
        roi_tensor = transform(roi_color).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(roi_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_labels[predicted.item()]

        # Draw results on original BGR frame
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
