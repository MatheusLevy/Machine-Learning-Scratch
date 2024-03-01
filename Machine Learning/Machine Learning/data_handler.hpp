#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

class Data_Handler {
	std::vector<Data *> *data_array; // all data not splited
	std::vector<Data*> *training_data;
	std::vector<Data*> *validation_data;
	std::vector<Data*> *test_data;
	
	int num_classes;
	int feature_vector_size;
	std::map<uint8_t, int> class_map;

	const double TRAIN_SET_PERCENTE = 0.75;
	const double TEST_SET_PERCENT = 0.20;
	const double VALIDATION_SET_PERCENT = 0.05;

	public:
		Data_Handler();
		~Data_Handler();

		void read_feature_vector(std::string path);
		void read_feature_labels(std::string path);
		void split_data();
		void count_classes();

		uint32_t convert_to_little_endian(const unsigned char *bytes);

		std::vector<Data*> * get_training_data();
		std::vector<Data*> * get_validation_data();
		std::vector<Data*> * get_testing_data();
};
#endif