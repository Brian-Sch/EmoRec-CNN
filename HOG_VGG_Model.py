import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2

from skimage.feature import local_binary_pattern
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch.utils.tensorboard import SummaryWriter

import kagglehub

# Download latest version
try:
    data_dir = kagglehub.dataset_download("msambare/fer2013")
except:
    data_dir = r'C:\Users\USER\PycharmProjects\Brian_Home\FER2013'  # Modify this path to the location of your FER2013 folder

print("Path to dataset files:", data_dir)

writer = SummaryWriter('runs/FER2013_Train_v42') # command: 'tensorboard --logdir=/path/to/logs --bind_all' in python Terminal, it should give you a url.

layout = {
    "Loss/Accuracy": {
        "loss": ["Multiline", ["Train/Loss", "Validation/Loss"]],
        "accuracy": ["Multiline", ["Train/Accuracy", "Validation/Accuracy"]],
    },
}
writer.add_custom_scalars(layout)


torch.backends.cudnn.enabled = False

# Mapping of numeric labels to emotion names
label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def labels_to_names(labels):
    return [label_map[label.item()] for label in labels]

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
        try:
            image_np = np.array(img.convert('L'))
        except:
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
        try:
            image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        except:
            image_color = image_np
        image_with_kps = cv2.drawKeypoints(image_color, keypoints, None, color=(0, 255, 0), flags=0)

        return torch.from_numpy(combined_features), lbp_image, image_with_kps

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

        # Fully connected layers
        x = self.pool1d(F.relu(self.bn7(self.fc1(x))))
        x = self.dropout4(x)
        x = self.pool1d(F.relu(self.bn8(self.fc2(x))))
        # x = self.pool1d(F.relu(self.bn8(self.fc3(x))))
        x = self.dropout4(x)
        return F.log_softmax(self.fc4(x), dim=1)

def plot_to_tensorboard(cm, class_names):
    """
    Plot the confusion matrix as an image in TensorBoard.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title="Normalized Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2%'  # Format as percentage
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def show_images(data_loader, data_set):
    # Fetch the first batch of images and labels
    dataiter = iter(data_loader)
    images, labels = next(dataiter)


    # Get class names from the dataset
    classes = data_set.classes

    num_images=6

    images = images[:num_images]  # Select the first `num_images` from the batch
    labels = labels[:num_images]
    fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=num_images)
    for i, ax in enumerate(axes):
        img = images[i] * 0.5 + 0.5  # unnormalize
        img = img.numpy().transpose((1, 2, 0))# Convert from Tensor image
        # img = lbp.forward(img) # local binary pattern
        img = np.clip(img, 0, 1)
        ax.imshow(img.squeeze(), cmap='gray')  # Assuming grayscale images
        ax.title.set_text('Class: ' + classes[labels[i]])
        ax.axis('off')
    plt.show()

def imshow(img, title=None):
    img = img.cpu().numpy()  # Move tensor to CPU and convert to numpy
    img = np.squeeze(img)  # Remove the channel dimension for grayscale images
    img = img / 2 + 0.5  # Unnormalize
    plt.imshow(img, cmap='gray')  # Use grayscale color map
    if title is not None:
        plt.title(title)
    plt.show()

def load_data(data_dir, batch_size=64):

    # Create datasets
    # train_dataset = datasets.ImageFolder(root=f"{data_dir}/train",
    #                                      transform=transforms.Compose([
    #                                                                    transforms.Grayscale(),
    #                                                                    transforms.Resize((48, 48)),
    #                                                                    transforms.ToTensor(),
    #                                                                    transforms.Normalize(0.449, 0.226)]))
    #
    # test_dataset = datasets.ImageFolder(root=f"{data_dir}/test",
    #                                     transform=transforms.Compose([
    #                                                                   transforms.Grayscale(),
    #                                                                   transforms.Resize((48, 48)),
    #                                                                   transforms.ToTensor(),
    #                                                                   transforms.Normalize(0.449, 0.226)]))
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train",
                                         transform=transforms.Compose([
                                                                       transforms.Grayscale(),
                                                                       transforms.Resize((48, 48)),
                                                                       transforms.RandomHorizontalFlip(p=0.5),
                                                                       transforms.RandomVerticalFlip(p=0.5),
                                                                       transforms.ToTensor()]))

    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test",
                                        transform=transforms.Compose([
                                                                      transforms.Grayscale(),
                                                                       transforms.Resize((48, 48)),
                                                                       transforms.ToTensor()]))


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


    return train_loader, test_loader, train_dataset, test_dataset

def train_model(model, device, train_loader, test_loader, criterion, optimizer, writer, num_epochs):
    training_start_time = time.time()
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Set model to training mode
        model.train()

        # Initializing parameters
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        batch = 0

        # Defining transformations for images
        transform = transforms.Compose([transforms.Normalize(0.449, 0.226)])
        feature_ext = CombinedFeaturesTransform()
        feature_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((200, 200))])

        # Starting Training
        for i, (images, labels) in enumerate(train_loader):
            batch += 1
            # image to GPU
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)


            # Prepare features for each image in the batch
            feature_vectors, lbp_images, kps_images = zip(
                *[feature_ext(feature_transform(image)) for image in images])

            # Convert feature_vectors to a tensor if not already
            feature_vectors = torch.stack(feature_vectors).to(device)

            # PreProcess image for CNN
            images = transform(images)

            # Forward pass
            outputs = model(images, feature_vectors)

            # Predicted Results
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # Backwards and optimize
            optimizer.zero_grad()


            loss.backward()
            optimizer.step()

            # Updating and logging Loss\Accuracy graphs
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
            if (i + 1) % 100 == 0 or i == len(train_loader) - 1:
                print(f'            Batch: {batch} Loss: {loss.item()}')
                # Log to TensorBoard
                num_images_to_log = 6  # How many images to log per batch
                emotion_names = labels_to_names(labels[:num_images_to_log])
                writer.add_images('Train/Images', images[:num_images_to_log], epoch * len(train_loader) + i)
                writer.add_text('Train/Labels', str(emotion_names), epoch * len(train_loader) + i)
                writer.add_text('Train/Predictions', str(labels_to_names(predicted[:num_images_to_log])), epoch * len(train_loader) + i)
                # Convert LBP and keypoints images to tensors and log them
                lbp_tensors = torch.stack([TF.to_tensor(lbp_image) for lbp_image in lbp_images])
                kps_tensors = torch.stack([TF.to_tensor(kps_image) for kps_image in kps_images])

                writer.add_images('Train/LBP Images', lbp_tensors[:num_images_to_log], epoch * len(train_loader) + i)
                writer.add_images('Train/Keypoint Images', kps_tensors[:num_images_to_log], epoch * len(train_loader) + i)
        # Calculating Epoch loss
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        print("     -Training dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        # writer.add_scalar('Loss', epoch_loss, epoch)
        # writer.add_scalar('Accuracy', epoch_acc, epoch)

        val_acc, cm = val_model(model, device, test_loader, epoch, writer)

        if val_acc > best_val_acc:
            # Save best acc model
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_state_dict.pth')
            print(f"Saved better model with accuracy: {best_val_acc:.4f}")
            # Plot Confusion matrix
            fig = plot_to_tensorboard(cm, list(label_map.values()))
            writer.add_figure('Confusion Matrix', fig, epoch)

    training_end_time = time.time()
    print(f"\nTraining time: {(training_end_time - training_start_time) / 60:.2f} minutes")
    print("Finished Training")

def val_model(model, device, test_loader, epoch, writer):
    # Validation phase
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    predicted_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    transform = transforms.Compose([transforms.Normalize(0.449, 0.226)])
    feature_ext = CombinedFeaturesTransform()
    feature_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((200, 200))])
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # Prepare features for each image in the batch
            # batch_features = [feature_ext(transforms.ToPILImage()(img)) for img in images]
            # features = torch.stack(batch_features).to(device)  # Stack features and move to device

            feature_vectors, lbp_images, kps_images = zip(
                *[feature_ext(feature_transform(image)) for image in images])
            # Convert feature_vectors to a tensor if not already
            feature_vectors = torch.stack(feature_vectors).to(device)

            images = transform(images)

            # Forward pass
            outputs = model(images, feature_vectors)

            # Predicted Results
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            predicted_correct += (predicted == labels).sum().item()

            # Load results to cpu for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Optionally log images periodically (e.g., at the end of validation)
            if (i + 1) % 100 == 0 or i == len(train_loader) - 1:  # Log images at the end of each epoch
                # Log to TensorBoard
                num_images_to_log = 6  # How many images to log
                emotion_names = labels_to_names(labels[:num_images_to_log])
                writer.add_images('Validation/Images', images[:num_images_to_log], epoch * len(test_loader) + i)
                writer.add_text('Validation/Labels', str(emotion_names), epoch * len(test_loader) + i)
                writer.add_text('Validation/Predictions', str(labels_to_names(predicted[:num_images_to_log])),
                                epoch * len(test_loader) + i)

                # Convert LBP and keypoints images to tensors and log them
                lbp_tensors = torch.stack([TF.to_tensor(lbp_image) for lbp_image in lbp_images])
                kps_tensors = torch.stack([TF.to_tensor(kps_image) for kps_image in kps_images])
                writer.add_images('Validation/LBP Images', lbp_tensors[:num_images_to_log], epoch * len(train_loader) + i)
                writer.add_images('Validation/Keypoint Images', kps_tensors[:num_images_to_log], epoch * len(train_loader) + i)


    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100.00 * predicted_correct / total
    print("     -Testing dataset. Got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f"
          % (predicted_correct, total, epoch_acc, epoch_loss))
    writer.add_scalar('Validation/Loss', epoch_loss, epoch)
    writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)
    # writer.add_scalar('Loss', epoch_loss, epoch)
    # writer.add_scalar('Accuracy', epoch_acc, epoch)
    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize

    return epoch_acc, cm


if __name__ == '__main__':

    # Hyper-parameters
    input_size = 48*48 # 48x48
    num_classes = 7
    num_epochs = 60
    batch_size = 16 # 32
    learning_rate = 0.001

    # init the CNN model
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ConvNet(num_classes=num_classes).to(device)
    model = ConvNet().to(device)

    # Loss Function Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loading Data
    train_loader, test_loader, train_dataset, test_dataset = load_data(data_dir, batch_size)

    train_model(model, device, train_loader, test_loader, criterion, optimizer, writer, num_epochs)

    writer.close()
    print("\nFinish")

