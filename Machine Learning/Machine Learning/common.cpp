#include "common.hpp"

void common_data::set_training_data(std::vector<Data*>* vect) {
	training_data = vect;
}
void common_data::set_test_data(std::vector<Data*>* vect) {
	test_data = vect;
}
void common_data::set_valiation_data(std::vector<Data*>* vect) {
	validation_data = vect;
}