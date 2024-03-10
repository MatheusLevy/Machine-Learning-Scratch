#include <iostream>
#include <vector>
#include "data_handler.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

/*
int main()
{
    Data_Handler* dh = new Data_Handler();
    dh->read_feature_vector("../../mnist/train-images.idx3-ubyte");
    dh->read_feature_labels("../../mnist/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
    Data* data = dh->get_data()->at(2);
    auto features = data->get_feature_vector();
    printf("\n Size of Features: %u", features->size());
    cv::Mat image(28, 28, CV_8UC1);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                image.at<uint8_t>(i, j) = data->get_feature_vector()->at(i * 28 + j);
            }
        }
    imshow("image", image);
    waitKey();
    return 0;
}
*/
