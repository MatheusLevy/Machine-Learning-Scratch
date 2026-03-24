#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

Eigen::VectorXd Mat2Vector(const cv::Mat& img) {
    Eigen::VectorXd vec(img.rows * img.cols);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            vec(i * img.cols + j) = static_cast<double>(img.at<uchar>(i, j));
        }
    }
    return vec;
}

Eigen::MatrixXd patchImage(const cv::Mat& img, const int patchSize) {
    Eigen::MatrixXd patches = Eigen::MatrixXd::Zero((img.rows / patchSize) * (img.cols / patchSize), patchSize * patchSize);

    for (int i=0; i <= img.rows - patchSize; i += patchSize){
        for (int j=0; j <= img.cols - patchSize; j += patchSize){
           int patchIdx = (i / patchSize) * (img.cols / patchSize) + (j / patchSize);
           cv::Rect roi_patch(j, i, patchSize, patchSize);
           cv::Mat roi = img(roi_patch);
           Eigen::VectorXd patchVec = Mat2Vector(roi);
           patches.row(patchIdx) = patchVec;
        }
    }
    return patches;
}

int main() {
    cv::Mat img1 = cv::imread("frame1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread("frame2.jpg", cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Erro: Nao foi possivel carregar as imagens frame1.jpg e/ou frame2.jpg!" << std::endl;
        return -1;
    }

    Eigen::MatrixXd patches1 = patchImage(img1, 8);
    std::cout << "Shape of patches1:\n" << patches1.rows() << " x " << patches1.cols() << std::endl;
    return 0;
}