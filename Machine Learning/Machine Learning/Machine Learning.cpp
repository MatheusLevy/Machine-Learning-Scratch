#include <iostream>
#include <vector>
#include "data_handler.hpp"

int main()
{
    Data_Handler* dh = new Data_Handler();
    dh->read_feature_vector("../../mnist/train-images.idx3-ubyte");
    dh->read_feature_labels("../../mnist/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    return 0;
}
