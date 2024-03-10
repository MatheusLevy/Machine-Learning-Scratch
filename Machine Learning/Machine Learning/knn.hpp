#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "data.hpp"

class knn {
	int k;
	std::vector<Data*>* neighbors;
	std::vector<Data*>* training_data;
	std::vector<Data*>* test_data;
	std::vector<Data*>* validation_data;

	public:
		knn(int);
		knn();
		~knn();

		void find_knearest(Data* query_point);
		void set_training_data(std::vector<Data*>* vect);
		void set_test_data(std::vector<Data*>* vect);
		void set_valiation_data(std::vector<Data*>* vect);
		void set_k(int val);

		int predict();
		double calculate_distance(Data* query_point, Data* input);
		double validate_performance();
		double test_performance();
};
#endif 
