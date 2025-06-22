import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from EmoRec_Train import ConvNet, CombinedFeaturesTransform

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

if __name__ == '__main__':

    # Mapping of numeric labels to emotion names
    label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


    # Initialize transformation and model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((48, 48)),  # Resize the image to match the model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(0.449, 0.226)
    ])
    feature_transform = CombinedFeaturesTransform()

    # Load model
    model = ConvNet()
    model.load_state_dict(torch.load('best_model_state_dict.pth'))
    model.eval()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            lbp_image = None
            image_with_kps = None

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]

                input_tensor = transform(face_roi)
                input_tensor = input_tensor.unsqueeze(0)
                features, lbp_image, image_with_kps = feature_transform(face_roi)
                features_tensor = features.unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor, features_tensor)
                    predicted_class = torch.argmax(output, dim=1)
                    emotion = label_map[predicted_class.item()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Safely show images only if they are defined
            if lbp_image is not None:
                cv2.imshow('LBP Image', lbp_image)
            if image_with_kps is not None:
                cv2.imshow('Keypoints', image_with_kps)

            cv2.imshow('Original', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()