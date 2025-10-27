# 🤖 Machine Learning & Deep Learning from Scratch

A complete and organized repository with Machine Learning and Deep Learning algorithm implementations from scratch, practical projects, and educational notebooks.

## 📚 Repository Structure

This repository is organized into three main areas:

### 🔷 1. Machine Learning (Classical Algorithms)
Implementations of fundamental machine learning algorithms.

#### 📊 Supervised Learning
- **Regression**
  - Linear Regression (Gradient Descent & Normal Equation)
  - Locally Weighted Linear Regression
  - Softmax Regression
  
- **Classification**
  - Logistic Regression (Gradient Descent & Newton's Method)
  - Naive Bayes

#### 🔍 Unsupervised Learning
- **Clustering**
  - K-Means (C++ implementation)
  
#### 🎯 Nearest Neighbors
- K-Nearest Neighbors (KNN) - C++ implementation

#### ⚙️ C++ Implementations
Low-level C++ implementations for optimized performance.

---

### 🔷 2. Deep Learning (Neural Networks and Deep Learning)

#### 🧠 Fundamentals
- Basic neural network implementation
- Layers: Input, Hidden, Output
- Neurons and network architecture

#### 👁️ Computer Vision

**Classic CNN Architectures:**
- LeNet
- VGG16
- GoogLeNet (Inception)
- ResNet
- EfficientNet

**Practical Projects:**
- Melanoma Classification
- Skin Cancer Classification
- FP16 Optimization

**OpenCV:**
- Basic tutorials: filters, colors, contours, edge detection
- Projects: Color detection, Face anonymization
- Various experiments

**Utilities:**
- Data loaders
- Mean and standard deviation calculation
- Binary Focal Loss
- Helper functions

#### 🎨 Generative Models
- Autoencoders
- Fully Connected GAN
- DCGAN (Deep Convolutional GAN)

#### 💬 NLP (Natural Language Processing)

**Fundamentals:**
- N-grams
- Data gathering and preprocessing

**Sequence Models:**
- RNN (Recurrent Neural Networks)
- RNN from Scratch
- LSTM (Long Short-Term Memory)
- Sequence-to-Sequence
- Sequence-to-Sequence with Attention

**Transformers:**
- Transformers from Scratch
- GPT-1 from Scratch
- Transformers for Translation
- Project: GPT trained on Wikipedia articles

**Applications:**
- Spam Classifier (Naive Bayes)
- Name Generator
- Datasets and metrics

#### 🎓 Specialized Techniques
- Transfer Learning
- Multi-label Classification
- Transpose Convolution

---

### 🔷 3. Recommendation Systems
- Popularity-based recommendation system

---

## 📂 Datasets

All datasets are centralized in the `data/` folder:
- **images/**: MNIST and other images
- **tabular/**: Iris.csv and other tabular datasets

---

## 🚀 How to Use

### Prerequisites

**Python:**
```bash
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision
pip install tensorflow
pip install opencv-python
pip install nltk
```

**C++ (for native implementations):**
- C++ Compiler (g++, MSVC, etc.)
- CMake (optional)

### Running Notebooks

1. Install Jupyter:
```bash
pip install jupyter notebook
```

2. Navigate to desired folder:
```bash
cd 2-Deep-Learning/computer_vision/architectures
jupyter notebook
```

### Compiling C++ Code

```bash
cd 1-Machine-Learning/cpp_implementations
g++ -std=c++17 -o program main.cpp knn.cpp kmeans.cpp data_handler.cpp
./program
```

---

## 📖 Topics by Area

### Classical Machine Learning
- Linear and Logistic Regression
- Distance-based algorithms (KNN)
- Probabilistic models (Naive Bayes)
- Clustering (K-Means)

### Deep Learning - Computer Vision
- Fundamental CNN architectures
- Transfer Learning
- GANs and generative models
- Medical projects (skin cancer detection)
- OpenCV and image processing

### Deep Learning - NLP
- Classical language models (N-grams)
- RNNs and LSTMs
- Attention Mechanisms
- Transformers and GPT
- Practical applications

---

## 🎯 Repository Goals

1. **Educational**: Didactic implementations from scratch
2. **Practical**: Real projects and applications
3. **Organized**: Clear structure by topics and subtopics
4. **Complete**: From fundamentals to advanced techniques

---

## 🛠️ Technologies Used

- **Languages**: Python, C++
- **DL Frameworks**: PyTorch, TensorFlow
- **Computer Vision**: OpenCV
- **NLP**: NLTK, Transformers
- **Data**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

---

## 📝 Detailed Directory Structure

```
Machine-Learning-Scratch/
├── data/                                      # Centralized datasets
│   ├── images/                               # MNIST, etc.
│   └── tabular/                              # CSV files
│
├── 1-Machine-Learning/                        # Classical ML
│   ├── supervised/
│   │   ├── regression/                       # Regression algorithms
│   │   ├── classification/                   # Classification algorithms
│   │   └── examples/                         # Practical examples
│   ├── unsupervised/
│   │   ├── clustering/                       # K-means
│   │   └── dimensionality_reduction/         # PCA, etc.
│   ├── nearest_neighbors/                    # KNN
│   └── cpp_implementations/                  # C++ implementations
│
├── 2-Deep-Learning/                           # Deep Learning
│   ├── fundamentals/                         # Basic neural networks
│   │   └── layers/                           # Layer implementations
│   ├── computer_vision/
│   │   ├── architectures/                    # Classic CNNs
│   │   ├── projects/                         # Practical projects
│   │   ├── opencv/                           # OpenCV
│   │   └── utils/                            # Utilities
│   ├── generative_models/                    # GANs, Autoencoders
│   ├── nlp/
│   │   ├── fundamentals/                     # N-grams, preprocessing
│   │   ├── sequence_models/                  # RNN, LSTM
│   │   ├── transformers/                     # Transformers, GPT
│   │   └── applications/                     # Practical projects
│   └── specialized/                          # Specialized techniques
│
├── 3-Recommendation-Systems/                  # Recommendation systems
│
└── README.md                                  # This file
```

---

## 🤝 Contributions

This is a personal study repository, but suggestions and improvements are welcome!

---

## 📄 License

This project is for educational purposes.

---

## 📧 Contact

- GitHub: [@MatheusLevy](https://github.com/MatheusLevy)

---

**Last updated**: October 2025