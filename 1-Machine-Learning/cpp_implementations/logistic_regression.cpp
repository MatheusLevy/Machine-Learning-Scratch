#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

using namespace Eigen;

const double LR = 0.01;

double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

VectorXd batch_predict(const MatrixXd& X, const Vector2d& w) {
    VectorXd predictions(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        predictions(i) = sigmoid(X.row(i).dot(w));
    }
    return predictions;
}

int predict(const Vector2d& x, const Vector2d& w) {
    return sigmoid(x.dot(w)) > 0.5 ? 1 : 0;
}

double gradient_j(const MatrixXd& X, const VectorXd& y, const Vector2d& w) {
    VectorXd h = batch_predict(X, w);
    double loss = 0.0;
    for (int i = 0; i < y.size(); ++i) {
        loss += -y(i) * std::log(h(i)) - (1 - y(i)) * std::log(1 - h(i));
    }
    return loss / y.size();
}

Vector2d gradient_ascend_step(const MatrixXd& X, const VectorXd& y, const Vector2d& w) {
    VectorXd h = batch_predict(X, w);
    VectorXd error = h - y;
    Vector2d grad = X.transpose() * error / y.size();
    return w - LR * grad;
}

Vector2d fit(const MatrixXd& X, const VectorXd& y, int epochs = 1000) {
    Vector2d w = Vector2d::Random(); // Initialize weights randomly
    std::cout << "Epoch\tLoss" << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        w = gradient_ascend_step(X, y, w);
        double loss = gradient_j(X, y, w);
        if (epoch % 100 == 0) {
            std::cout << epoch << "\t" << loss << std::endl;
        }
    }
    return w;
}

double compute_loss(const MatrixXd& X, const VectorXd& y, const Vector2d& w) {
    return gradient_j(X, y, w);
}

int main() {
    // Dataset: 6 samples, 2 features (age, hours_studied), 3 class 0, 3 class 1
    MatrixXd X(6, 2);
    X << 20, 5,
         25, 3,
         30, 4,
         22, 8,
         27, 9,
         32, 10;

    VectorXd y(6);
    y << 0, 0, 0, 1, 1, 1;

    Vector2d w = fit(X, y, 10000);


    Vector2d new_sample(24, 7);
    int prediction = predict(new_sample, w);
    std::cout << "Prediction for age 24, hours 7: " << prediction << std::endl;

    return 0;
}