#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"


class Data {
	std::vector<uint8_t> * feature_vector; // No class at end
	std::vector<double> * double_feature_vector; 
	std::vector<double>* normalized_feature_vector;
	std::vector<int>* class_vector;
	uint8_t label;
	int enum_label;
	double distance;

	public:
		Data();
		~Data();
		void set_double_feature_vector(std::vector<double> *);
		void set_feature_vector(std::vector<uint8_t> *);
		void append_to_feature_vector(uint8_t);
		void append_to_double_feature_vector(double);
		void set_label(uint8_t);
		void set_class_vector(int count);
		void set_enumerated_label(int);
		void set_distance(double val);
		void set_normalized_feature_vector(std::vector<double> *);
		
		int get_feature_vector_size();
		int get_double_feature_vector_size();
		uint8_t get_label();
		uint8_t get_enumerated_label();
		double get_distance();
		std::vector<uint8_t>* get_feature_vector();
		std::vector<double>* get_double_feature_vector();
		std::vector<int>* get_class_vector();
		std::vector<double>* get_normalized_feature_vector();
};		
#endif 
