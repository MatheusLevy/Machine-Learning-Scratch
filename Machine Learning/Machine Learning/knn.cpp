#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"

knn::knn(int val) {
	k = val;
}

knn::knn() {
}

knn::~knn() {
}

void knn::set_k(int val) {
	k = val;
}

void knn::find_knearest(Data* query_point) {
	neighbors = new std::vector<Data*>;
	double min = std::numeric_limits<double>::max();
	double previous_min = min;
	int index = 0;
	for (int i = 0; i < k; i++) {
		if (i == 0) {
			for (int j = 0; j < training_data->size(); j++) {
				double distance = calculate_distance(query_point, training_data->at(j));
				training_data->at(j)->set_distance(distance);
				if (distance < min) {
					min = distance;
					index = j;
				}
			}
			neighbors->push_back(training_data->at(index));
			previous_min = min;
			min = std::numeric_limits<double>::max();
		}
		else {
			for (int j = 0; j < training_data->size(); j++) {
				double distance = training_data->at(j)->get_distance();
				if (distance > previous_min && distance < min) {
					min = distance;
					index = j;
				}
			}
			neighbors->push_back(training_data->at(index));
			previous_min = min;
			min = std::numeric_limits<double>::max();
		}
	}
}

int knn::predict() {
	std::map<uint8_t, int> class_freq;
	for (int i = 0; i < neighbors->size(); i++) {
		if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end()) {
			class_freq[neighbors->at(i)->get_label()] = 1;
		}
		else {
			class_freq[neighbors->at(i)->get_label()]++;
		}
	}
	int best = 0;
	int max = 0;
	for (auto kv : class_freq) {
		if (kv.second > max) {
			max = kv.second;
			best = kv.first;
		}
	}
	neighbors->clear();
	return best;
}

double knn::calculate_distance(Data* query_point, Data* input) {
	double distance = 0.0;
	if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
		printf("Error vector size missmatch");
	}
	for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++) {
		distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
	}
	distance = sqrt(distance);
	return distance;
}

double knn::validate_performance() {
	double current_perfomance = 0;
	int count = 0;
	int data_index = 0;
	for (Data* query_point : *validation_data) {
		find_knearest(query_point);
		int prediction = predict();
		if (prediction == query_point->get_label()) {
			count++;
		}
		data_index++;
		printf("Current perfomance = %.3f %%\n", ((double)count*100 / (double)data_index));
	}
	current_perfomance = ((double)count * 100 / (double)validation_data->size());
	printf("Validation Perfomance = %.3f %%\n", current_perfomance);
	return current_perfomance;
}

double knn::test_performance() {
	double current_performace = 0;
	int count = 0;
	for (Data* query_point : *test_data) {
		find_knearest(query_point);
		int prediction = predict();
		if (prediction == query_point->get_label()) {
			count++;
		}
	}
	current_performace = (double)count * 100 / test_data->size();
	printf("Tested Perfomance for K = %d : %.3f %%", k, current_performace);
	return current_performace;
}

int main() {
	Data_Handler* dh = new Data_Handler();
	dh->read_feature_vector("../../mnist/train-images.idx3-ubyte");
	dh->read_feature_labels("../../mnist/train-labels.idx1-ubyte");
	dh->split_data();
	dh->count_classes();
	knn* knearest = new knn();
	knearest->set_training_data(dh->get_training_data());
	knearest->set_valiation_data(dh->get_validation_data());
	knearest->set_test_data(dh->get_testing_data());
	double performance = 0;
	double best_performance = 0;
	int best_k = 1;
	knearest->set_k(1);
	knearest->validate_performance();
}