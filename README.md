# Project: Convolutional Neural Networks and Face Verification with CNNs

## Section 1: Building Convolutional Neural Networks (CNNs) from Scratch

In this project, a CNN framework was built entirely from scratch, focusing on implementing core components such as:

- **Convolutional Layers:** Designed Conv1D and Conv2D layers with stride variations, enabling flexible convolution operations.
- **Pooling Mechanisms:** Implemented MaxPooling and MeanPooling layers to reduce spatial dimensions and computational overhead.
- **Resampling Layers:** Created upsampling and downsampling mechanisms to simulate stride operations effectively.
- **Flatten and Linear Layers:** Integrated fully connected layers for classification tasks following feature extraction.
- **End-to-End Training Pipeline:** Achieved forward and backward propagation for training CNNs using only NumPy, enabling complete control over the learning process.
- **Converted Scanning MLPs to CNNs:** Converted Simple and Distributed Scanning MLPs to fully functional CNN models.

This section demonstrates a detailed understanding of CNN architectures, emphasizing the mathematical foundations essential for deep learning from the ground up.

---

## Section 2: Face Classification and Verification Using CNNs

This section focuses on building a robust face recognition system using CNNs for feature extraction and classification, followed by verification through similarity metrics.

### Highlights:
1. **Face Embedding Creation:**
   - Developed CNN architectures to encode distinctive facial features into fixed-length vectors (face embeddings).
   - Used these embeddings for classification and verification tasks.

2. **Loss Functions:**
   - Employed Softmax Cross-Entropy Loss for classification tasks.
   - Experimented with advanced margin-based loss functions like ArcFace to improve feature separability.

3. **Verification Pipeline:**
   - Extracted feature vectors from face images and computed similarity scores between pairs for verification.

4. **Architectures:**
   - Explored architectures such as ResNet, SeNet, and EfficientNet for robust feature extraction and classification.

5. **Data Augmentation and Regularization:**
   - Experimented with various augmentation techniques such as CutMix, MixUp, and RandomPerspective.
   - Applied regularization techniques like Label Smoothing and DropBlock to enhance generalization.
   - Used MixedPrecision Training to speed up the training process and optimize computational efficiency.

6. **Performance:**
   - Achieved the highest verification accuracy of EER **8.6** using ResNet18 followed by fine-tuning with ArcFace Loss.
   - The dataset used for training and evaluation was sourced from ImageNet.

This section showcases the development of an end-to-end solution for face recognition, highlighting its application in real-world scenarios requiring accurate identity verification.
