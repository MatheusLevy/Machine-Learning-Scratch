#ifndef __KNN_H
#define __KNN_H

#include "common.hpp"

class knn : public common_data{
	int k;
	std::vector<Data*>* neighbors;

	public:
		knn(int);
		knn();
		~knn();

		void find_knearest(Data* query_point);
		void set_k(int val);

		int predict();
		double calculate_distance(Data* query_point, Data* input);
		double validate_performance();
		double test_performance();
};
#endif 
