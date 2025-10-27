# Machine Learning (Classical Algorithms)

This folder contains implementations of fundamental Machine Learning algorithms.

## 📂 Structure

### Supervised Learning

#### Regression
- **Linear Regression (Gradient Descent)**: Linear regression using gradient descent
- **Linear Regression (Normal Equation)**: Linear regression using normal equation
- **Locally Weighted Linear Regression**: Locally weighted linear regression
- **Softmax Regression**: Multinomial regression

#### Classification
- **Logistic Regression (Gradient Descent)**: Binary classification using gradient descent
- **Logistic Regression (Newton's Method)**: Binary classification using Newton's method
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

### Unsupervised Learning

#### Clustering
- **K-Means**: Clustering algorithm (C++ implementation)

### Nearest Neighbors
- **K-Nearest Neighbors (KNN)**: Proximity-based classification (C++ implementation)

### C++ Implementations
Low-level C++ implementations of the above algorithms, including:
- Data manipulation (`data_handler.cpp/hpp`)
- Data structures (`data.cpp/hpp`)
- Common utilities (`common.cpp/hpp`)

## 🚀 How to Use

### Python Algorithms
```python
# Basic usage example
python Linear_Regression_Gradient_Descent.py
```

### C++ Algorithms
```bash
cd cpp_implementations
g++ -std=c++17 -o ml_program *.cpp
./ml_program
```

## 📖 Recommended Study Order

1. Linear Regression (Normal Equation) - simpler
2. Linear Regression (Gradient Descent) - introduces optimization
3. Logistic Regression - binary classification
4. Naive Bayes - probabilistic model
5. K-Nearest Neighbors - instance-based algorithm
6. K-Means - unsupervised learning
