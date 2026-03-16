#include <iostream>
#include <vector>
#include <algorithm>


double LR = 0.01; 

/*
    theta_j = theta_j - lr * (error)*xj

*/
double gradient_j(std::vector<double>& pred, std::vector<double>& y, std::vector<std::vector<double>>& X, int j){
    double sum = 0;
    double error;
    for (int i=0; i<pred.size(); i++){
        error = pred[i] - y[i];
        if (j == 0) {
            sum += error; // x0 is always 1 for the intercept term
        } else {
            sum += error * X[j-1][i]; // X[j-1] because X is 0-indexed and j starts from 1
        }
    }
    return sum / pred.size();
}

void gradient_descent(std::vector<std::vector<double>>& X, std::vector<double>& y,
                      std::vector<double>& pred, std::vector<double>& params){
    for (int j=0; j<params.size(); j++){
        params[j] -= LR * gradient_j(pred, y, X, j);
    }
}

double normalize_min_max(double value, double min, double max) {
    return (value - min) / (max - min);
}

class LinearRegression {
    public:
        std::vector<double> coefficients;
        
        LinearRegression(int number_of_features) {
            coefficients.resize(number_of_features + 1, 0.0);
        };

        std::vector<double> predict(std::vector<std::vector<double>>& X) {
            std::vector<double> preds(X[0].size(), 0.0);
            for (int i=0; i<X[0].size(); i++){
                double pred = coefficients[0]; // Intercept term
                for (int j=0; j<X.size(); j++){
                    pred += coefficients[j+1] * X[j][i];
                }
                preds[i] = pred;
            }
            return preds;
        }

        void fit(std::vector<std::vector<double>>& X, std::vector<double>& y, int iterations) {
            std::vector<double> pred(X[0].size(), 0.0);
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

int main(){
    // House data: features in rows, samples in columns
    // X = {ft² across all samples, bedrooms across all samples}
    std::vector<std::vector<double>> X = {
        {1200, 1500, 1800},  // ft² for each sample
        {2, 3, 3}             // bedrooms for each sample
    };
    std::vector<double> y = {200000, 250000, 300000};

    // Normaliza cada feature (linha) com min-max
    for (int j = 0; j < X.size(); j++) {
        double min = *std::min_element(X[j].begin(), X[j].end());
        double max = *std::max_element(X[j].begin(), X[j].end());
        for (int i = 0; i < X[j].size(); i++) {
            X[j][i] = normalize_min_max(X[j][i], min, max);
        }
    }

    LinearRegression model(X.size());
    std::vector<double> predictions = model.predict(X);
    std::cout << "Initial predictions: ";
    for (double pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    model.fit(X, y, 100000);
    predictions = model.predict(X);
    std::cout << "Predictions after training: ";
    for (double pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    return 0;
}
