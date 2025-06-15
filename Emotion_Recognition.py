import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

# Feature extracture
class CombinedFeaturesTransform:
    def __init__(self, lbp_points=16, lbp_radius=2, orb_max_features=256, lbp_bins=256, max_descriptors=32,
                 descriptor_length=32):
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius
        self.orb_max_features = orb_max_features
        self.lbp_bins = lbp_bins
        self.max_descriptors = max_descriptors
        self.descriptor_length = descriptor_length

    def __call__(self, img):
        # image_np = np.array(img.convert('L'))
        image_np = img

        # Compute LBP histogram
        frame_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(frame_gray, self.lbp_points, self.lbp_radius, method='uniform')
        lbp_image = np.interp(lbp, (lbp.min(), lbp.max()), (0, 255)).astype(np.uint8)
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(self.lbp_bins + 1), density=True)
        lbp_hist = lbp_hist.astype('float32')

        # Normalize the histogram
        lbp_hist = np.clip(lbp_hist, 0, 1)

        # Ensure lbp_hist is of fixed length
        if len(lbp_hist) < self.lbp_bins:
            lbp_hist = np.pad(lbp_hist, (0, self.lbp_bins - len(lbp_hist)), 'constant')
        else:
            lbp_hist = lbp_hist[:self.lbp_bins]

        # Calculate ORB descriptors
        orb = cv2.ORB_create(nfeatures=self.orb_max_features)
        keypoints, descriptors = orb.detectAndCompute(image_np, None)

        # Handle descriptors
        if descriptors is None:
            descriptors = np.zeros((0, self.descriptor_length), dtype=np.float32)

        # Flatten and manage descriptor size
        descriptors = descriptors.flatten()
        if len(descriptors) < self.max_descriptors * self.descriptor_length:
            descriptors = np.pad(descriptors, (0, self.max_descriptors * self.descriptor_length - len(descriptors)),
                                 'constant')
        else:
            descriptors = descriptors[:self.max_descriptors * self.descriptor_length]

        # Combine LBP histogram and ORB descriptors
        combined_features = np.hstack((lbp_hist, descriptors)).astype('float32')

        # Visualize keypoints
        # image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        image_with_kps = cv2.drawKeypoints(image_np, keypoints, None, color=(0, 255, 0), flags=0)

        return torch.from_numpy(combined_features), lbp_image, image_with_kps

# CNN Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convolutional layers

        # Layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Layer 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Layer 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Feature Layers
        self.fc_f1 = nn.Linear(8448, 1028)  # Adjust the input features to match the output of last conv layer
        self.bn_f1 = nn.BatchNorm1d(1028)
        self.dropout_f1 = nn.Dropout(0.4)

        self.fc_f2 = nn.Linear(1280, 256)  # Adjust the input features to match the output of last conv layer
        self.bn_f2 = nn.BatchNorm1d(256)
        self.dropout_f2 = nn.Dropout(0.4)


        # Assuming the input size to the network is 48x48
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6 + 1280, 1028)  # Notice the additional 256 features
        self.bn7 = nn.BatchNorm1d(1028)
        self.dropout4 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(514, 254)
        self.bn8 = nn.BatchNorm1d(254)
        # self.fc3 = nn.Linear(254, 64)
        self.fc4 = nn.Linear(127, 7)
        self.pool1d = nn.MaxPool1d(2,2)

    def forward(self, x, features):
        # CNN
        #original
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        x = self.dropout3(x)
        x = self.flatten(x)

        # # Modified
        # x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.conv1(x))))))
        # # x = self.dropout1(x)
        # x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.conv3(x))))))
        # # x = self.dropout2(x)
        # x = self.pool3(F.relu(self.bn6(self.conv6(F.relu(self.conv5(x))))))
        # x = self.dropout3(x)
        # x = self.flatten(x)

        # Features Network
        features = self.flatten(features)
        # features = F.relu(self.bn_f1(self.fc_f1(features)))
        # features = self.dropout_f1(features)

        #features = F.relu(self.bn_f2(self.fc_f2(features)))
        #features = self.dropout_f2(features)

        # Concatenate CNN output with external features
        x = torch.cat((x, features), dim=1)  # Make sure features are correctly resized if needed

        x = self.pool1d(F.relu(self.bn7(self.fc1(x))))
        x = self.dropout4(x)
        x = self.pool1d(F.relu(self.bn8(self.fc2(x))))
        # x = self.pool1d(F.relu(self.bn8(self.fc3(x))))
        x = self.dropout4(x)
        return F.log_softmax(self.fc4(x), dim=1)


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

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Process each face found
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess the face for the model (assuming 'transform' and 'feature_transform' are already defined)
            input_tensor = transform(face_roi)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            features, lbp_image, image_with_kps = feature_transform(face_roi)
            features_tensor = features.unsqueeze(0)  # Convert to tensor and add batch dimension

            # Predict emotion
            with torch.no_grad():
                output = model(input_tensor, features_tensor)
                predicted_class = torch.argmax(output, dim=1)
                emotion = label_map[predicted_class.item()]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the emotion label
            cv2.putText(frame, emotion, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('LBP Image', lbp_image)
        cv2.imshow('Keypoints', image_with_kps)
        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()