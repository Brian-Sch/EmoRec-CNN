🧠 Emotion Recognition using CNN and Handcrafted Features
========================================================

This project implements a Convolutional Neural Network (CNN) combined with handcrafted features (LBP and ORB) to classify human emotions based on facial expressions. It is based on the FER2013 dataset and is built entirely in PyTorch.

> ⚡ Final model achieves up to ~65% validation accuracy on FER2013 — approaching SOTA results for this challenging dataset.

--------------------------------------------------------

🎯 Objective
------------

Build an accurate emotion recognition system using a hybrid deep learning + feature engineering pipeline, exploring:
- CNNs for spatial feature learning
- Handcrafted features (LBP, ORB) for robustness to lighting/rotation
- Feature fusion for final classification

🧠 Methodology
--------------

Dataset: FER2013 (https://www.kaggle.com/datasets/msambare/fer2013)

- 35,887 grayscale facial images (48x48 pixels)
- 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise

Preprocessing:
- Grayscale conversion
- Resize to 48x48
- RandomHorizontalFlip and RandomVerticalFlip
- Normalization: mean=0.449, std=0.226

🏗️ Model Architecture
----------------------

CNN:
- 3 Convolutional Blocks:
  - Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
- Flatten
- Feature fusion with handcrafted descriptors
- Fully connected layers:
  - Dense → BN → ReLU → Dropout → MaxPool1D
  - Final softmax classifier

Handcrafted Features:
- LBP: Local Binary Pattern for texture features
- ORB: Keypoint and rotation-invariant features

⚙️ Hyperparameters
-------------------

- Epochs: 60
- Batch Size: 16
- Learning Rate: 0.001
- Optimizer: Adam
- Input Size: 48x48
- LBP Points: 16, Radius: 2
- ORB Keypoints: 256, Descriptor Size: 32

📈 Training
------------

To train:
$ python EmoRec_Train.py

TensorBoard:
$ tensorboard --logdir=runs --bind_all

TensorBoard events saved at:
runs/FER2013_Train_v42

Best model saved at:
models/best_model_state_dict.pth

🧪 Results
-----------

- Best validation accuracy: ~65%
- Overfitting noticed after ~20 epochs
- Confusion matrix and accuracy logged in TensorBoard

📊 Emotion Labels
------------------

0: angry
1: disgust
2: fear
3: happy
4: neutral
5: sad
6: surprise

🧰 Dependencies
----------------

$ pip install -r requirements.txt

- torch
- torchvision
- scikit-image
- opencv-python
- matplotlib
- tensorboard

🔬 References
--------------

- https://doi.org/10.1088/1742-6596/1804/1/012202
- https://doi.org/10.1038/s41598-022-11173-0


📌 License
-----------

MIT License

🙌 Acknowledgements
--------------------

Inspired by academic literature and open-source work.
