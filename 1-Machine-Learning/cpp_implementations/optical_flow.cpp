#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

constexpr double LAMBDA_MIN_THRESHOLD = 1e-4;
constexpr double EIGEN_RATIO_THRESHOLD = 1e-2;

std::vector<cv::Mat> patchImage(const cv::Mat& img, const int patchSize, const int stride = 1) {
    std::vector<cv::Mat> patchVectors;

    for (int i=0; i <= img.rows - patchSize; i += stride){
        for (int j=0; j <= img.cols - patchSize; j += stride){
           cv::Rect roi_patch(j, i, patchSize, patchSize);
           cv::Mat roi = img(roi_patch);
           patchVectors.push_back(roi);
        }
    }
    return patchVectors;
}

cv::Mat Ix(const cv::Mat& img) {
    cv::Mat grad_x;
    cv::Sobel(img, grad_x, CV_64F, 1, 0, 3);
    return grad_x;
}

cv::Mat Iy(const cv::Mat& img) {
    cv::Mat grad_y;
    cv::Sobel(img, grad_y, CV_64F, 0, 1, 3);
    return grad_y;
}


std::vector<cv::Mat> It(const std::vector<cv::Mat>& patches1, const std::vector<cv::Mat>& patches2){
    std::vector<cv::Mat> grad_t_vec;
    for (size_t i = 0; i < patches1.size(); ++i){
        cv::Mat grad_t = patches2[i] - patches1[i];
        grad_t_vec.push_back(grad_t);
    }
    return grad_t_vec;
}   

std::vector<cv::Mat> meanGradients(const std::vector<cv::Mat>& grad1, const std::vector<cv::Mat>& grad2) {
    std::vector<cv::Mat> mean_grad;
    mean_grad.reserve(grad1.size());
    for (size_t i = 0; i < grad1.size(); ++i) {
        mean_grad.push_back((grad1[i] + grad2[i]) * 0.5);
    }
    return mean_grad;
}

void convertToEigen(const cv::Mat& cvMat, Eigen::MatrixXd& eigenMat) {
    eigenMat.resize(cvMat.rows, cvMat.cols);
    for (int i = 0; i < cvMat.rows; ++i) {
        for (int j = 0; j < cvMat.cols; ++j) {
            eigenMat(i, j) = cvMat.at<double>(i, j);
        }
    }
}

std::vector<Eigen::MatrixXd> convertVectorOfCvMatToEigen(const std::vector<cv::Mat>& cvMats) {
    std::vector<Eigen::MatrixXd> eigenMats;
    for (const cv::Mat& cvMat : cvMats) {
        Eigen::MatrixXd eigenMat;
        convertToEigen(cvMat, eigenMat);
        eigenMats.push_back(eigenMat);
    }
    return eigenMats;
}

Eigen::Matrix2d A_Matrix(const Eigen::MatrixXd& Ix, const Eigen::MatrixXd& Iy) {
    Eigen::Matrix2d A;
    A(0, 0) = Ix.array().square().sum();
    A(0, 1) = (Ix.array() * Iy.array()).sum();
    A(1, 0) = A(0, 1); // Simetria
    A(1, 1) = Iy.array().square().sum();
    return A;
}

Eigen::Vector2d b_Vector(const Eigen::MatrixXd& Ix, const Eigen::MatrixXd& Iy, const Eigen::MatrixXd& It) {
    Eigen::Vector2d b;
    b(0) = -(Ix.array() * It.array()).sum();
    b(1) = -(Iy.array() * It.array()).sum();
    return b;
}

Eigen::MatrixXd computeOpticalFlow(const std::vector<cv::Mat>& grad_x_mean_vec, const std::vector<cv::Mat>& grad_y_mean_vec, const std::vector<cv::Mat>& grad_t_vec) {
    std::vector<Eigen::MatrixXd> Ix = convertVectorOfCvMatToEigen(grad_x_mean_vec);
    std::vector<Eigen::MatrixXd> Iy = convertVectorOfCvMatToEigen(grad_y_mean_vec);
    std::vector<Eigen::MatrixXd> It = convertVectorOfCvMatToEigen(grad_t_vec);

    Eigen::MatrixXd flows(Ix.size(), 2);
    for (size_t i = 0; i < Ix.size(); ++i) {
        Eigen::Matrix2d A = A_Matrix(Ix[i], Iy[i]);
        Eigen::Vector2d b = b_Vector(Ix[i], Iy[i], It[i]);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(A);
        if (es.info() != Eigen::Success) {
            flows(i, 0) = 0.0;
            flows(i, 1) = 0.0;
            continue;
        }

        const Eigen::Vector2d evals = es.eigenvalues();
        const double lambda_min = evals(0);
        const double lambda_max = evals(1);
        const bool isIllConditioned =
            lambda_max <= 0.0 ||
            lambda_min < LAMBDA_MIN_THRESHOLD ||
            (lambda_min / lambda_max) < EIGEN_RATIO_THRESHOLD;

        if (isIllConditioned) {
            flows(i, 0) = 0.0;
            flows(i, 1) = 0.0;
            continue;
        }

        Eigen::Vector2d flow_vector = A.ldlt().solve(b);
        flows(i, 0) = flow_vector(0);
        flows(i, 1) = flow_vector(1);
    }
    return flows;
}

void visualizeOpticalFlowFromMatrix(const cv::Mat& img, 
                                     const Eigen::MatrixXd& flow,
                                     const int patchSize) {
    cv::Mat flow_viz = img.clone();
    flow_viz.convertTo(flow_viz, CV_8U);
    cv::cvtColor(flow_viz, flow_viz, cv::COLOR_GRAY2BGR);
    
    int patch_idx = 0;
    for (int i = 0; i <= img.rows - patchSize; i += patchSize) {
        for (int j = 0; j <= img.cols - patchSize; j += patchSize) {
            if (patch_idx >= flow.rows()) break;
            
            // Extrair (u, v) da matriz
            double u = flow(patch_idx, 0);
            double v = flow(patch_idx, 1);
            
            // Centro do patch
            cv::Point center(j + patchSize/2, i + patchSize/2);
            cv::Point end(center.x + u * 5, center.y + v * 5);  // amplificar
            
            // Magnitude para colorir
            double magnitude = std::sqrt(u*u + v*v);
            cv::Scalar color = magnitude > 1.0 ? 
                cv::Scalar(0, 0, 255) :  // vermelho (movimento rápido)
                cv::Scalar(0, 255, 0);   // verde (movimento lento)
            
            // Desenhar seta
            cv::arrowedLine(flow_viz, center, end, color, 2, cv::LINE_AA, 0, 0.3);
            cv::circle(flow_viz, center, 3, cv::Scalar(255, 255, 0), -1);  // ponto azul
            
            patch_idx++;
        }
    }
    
    cv::imwrite("optical_flow.png", flow_viz);  // salvar imagem
}

int main() {
    cv::Mat img1 = cv::imread("frame1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("frame2.jpg", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Erro: Nao foi possivel carregar as imagens frame1.jpg e/ou frame2.jpg!" << std::endl;
        return -1;
    }

    cv::GaussianBlur(img1, img1, cv::Size(5, 5), 1.0);
    cv::GaussianBlur(img2, img2, cv::Size(5, 5), 1.0);

    img1.convertTo(img1, CV_64F);
    img2.convertTo(img2, CV_64F);

    const int PATCH_SIZE = 20;

    cv::Mat grad_x_img1 = Ix(img1);
    cv::Mat grad_y_img1 = Iy(img1);
    cv::Mat grad_x_img2 = Ix(img2);
    cv::Mat grad_y_img2 = Iy(img2);

    std::vector<cv::Mat> patches1 = patchImage(img1, PATCH_SIZE);
    std::vector<cv::Mat> patches2 = patchImage(img2, PATCH_SIZE);
    std::cout << "Numero de patches na imagem 1: " << patches1.size() << std::endl;
    std::cout << "Numero de patches na imagem 2: " << patches2.size() << std::endl;
    std::vector<cv::Mat> grad_x_f1 = patchImage(grad_x_img1, PATCH_SIZE);
    std::vector<cv::Mat> grad_y_f1 = patchImage(grad_y_img1, PATCH_SIZE);
    std::vector<cv::Mat> grad_t = It(patches1, patches2);
    std::vector<cv::Mat> grad_x_f2 = patchImage(grad_x_img2, PATCH_SIZE);
    std::vector<cv::Mat> grad_y_f2 = patchImage(grad_y_img2, PATCH_SIZE);
    std::vector<cv::Mat> grad_x_mean = meanGradients(grad_x_f1, grad_x_f2);
    std::vector<cv::Mat> grad_y_mean = meanGradients(grad_y_f1, grad_y_f2);

    std::cout << "Numero de gradientes Ix (mean): " << grad_x_mean.size() << std::endl;
    std::cout << "Numero de gradientes Iy (mean): " << grad_y_mean.size() << std::endl;
    std::cout << "Numero de gradientes It: " << grad_t.size() << std::endl;
    Eigen::MatrixXd flow = computeOpticalFlow(grad_x_mean, grad_y_mean, grad_t);
    visualizeOpticalFlowFromMatrix(img1, flow, PATCH_SIZE);
    return 0;
}