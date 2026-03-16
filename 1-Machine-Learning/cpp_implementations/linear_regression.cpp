#include <iostream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Dense>

double LR = 0.01;

/*
    theta_j = theta_j - lr * (error)*xj

*/
double gradient_j(const Eigen::Ref<const Eigen::VectorXd>& pred, const Eigen::Ref<const Eigen::VectorXd>& y, const Eigen::Ref<const Eigen::MatrixX2d>& X, int j){
    double sum = 0;
    double error;
    for (int i=0; i<pred.size(); i++){
        error = pred[i] - y[i];
        if (j == 0) {
            sum += error; // x0 is always 1 for the intercept term
        } else {
            sum += error * X(i, j-1); // X(i, j-1): sample i, feature j-1
        }
    }
    return sum / pred.size();
}

void gradient_descent(const Eigen::Ref<const Eigen::MatrixX2d>& X, const Eigen::Ref<const Eigen::VectorXd>& y,
                      const Eigen::Ref<const Eigen::VectorXd>& pred, Eigen::Vector3d& params){
    for (int j=0; j<params.size(); j++){
        params[j] -= LR * gradient_j(pred, y, X, j);
    }
}

double normalize_min_max(double value, double min, double max) {
    return (value - min) / (max - min);
}

class LinearRegression {
    public:
        Eigen::Vector3d coefficients = Eigen::Vector3d::Zero();

        Eigen::VectorXd predict(const Eigen::Ref<const Eigen::MatrixX2d>& X) {
            Eigen::VectorXd preds(X.rows());
            for (int i=0; i<X.rows(); i++){
                double pred = coefficients[0]; // Intercept term
                for (int j=0; j<X.cols(); j++){
                    pred += coefficients[j+1] * X(i, j);
                }
                preds[i] = pred;
            }
            return preds;
        }

        void fit(Eigen::Matrix<double, 3, 2>& X, Eigen::Matrix<double, 3, 1>& y, int iterations) {
            Eigen::VectorXd pred;
            for (int iter=0; iter<iterations; iter++){
                pred = predict(X);
                gradient_descent(X, y, pred, coefficients);

                if (iter % 1000 == 0) {
                    double mse = 0;
                    for (int i = 0; i < pred.size(); i++) {
                        double error = pred[i] - y[i];
                        mse += error * error;
                    }
                    mse /= pred.size();
                    std::cout << "Iter " << iter << " | MSE: " << mse << std::endl;
                }
            }
        }
};

void normalize_features(Eigen::Matrix<double, 3, 2>& X) {
    for (int j = 0; j < X.cols(); j++) {
        double min = X.col(j).minCoeff();
        double max = X.col(j).maxCoeff();
        for (int i = 0; i < X.rows(); i++) {
            X(i, j) = (X(i, j) - min) / (max - min);
        }
    }
}

void normalize_labels(Eigen::Matrix<double, 3, 1>& y) {
    double min = y.minCoeff();
    double max = y.maxCoeff();
    for (int i = 0; i < y.size(); i++) {
        y(i) = (y(i) - min) / (max - min);
    }
}

int main(){
    Eigen::Matrix<double, 3, 2> x;
    x << 
        2, 120,
        3, 150,
        3, 180;
    normalize_features(x);
    std::cout << "Normalized features:\n" << x << std::endl;
    Eigen::Matrix<double, 3, 1> y;
    y << 200000, 250000, 300000;
    double y_min = y.minCoeff(), y_max = y.maxCoeff();
    normalize_labels(y);
    std::cout << "Normalized labels:\n" << y << std::endl;
    LinearRegression model;
    model.fit(x, y, 10000);
    Eigen::VectorXd predictions = model.predict(x);
    predictions = predictions.array() * (y_max - y_min) + y_min;
    std::cout << "Predictions:\n" << predictions << std::endl;
    return 0;
}
