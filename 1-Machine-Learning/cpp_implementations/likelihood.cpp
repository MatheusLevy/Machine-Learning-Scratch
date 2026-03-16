#include <iostream>
#include <cmath>

double quadratic_error(double y[], double pred[], int size);

int main(){
	double y[] = {100, 150, 200, 250};
	double pred[] = {110, 130, 205, 250};
	int n = sizeof(y) / sizeof(y[0]);
	std::cout << "size: " << n << std::endl;
	double error = quadratic_error(y, pred, n);
	std::cout << error;
	return 0;
}

double quadratic_error(double y[], double pred[], int size){
	double sum = 0;
	double error;
	for(int i=0; i<size; i++){
		std::cout << "y: " << y[i] << " pred: " << pred[i] << std::endl;
		error = y[i] - pred[i];
		sum += pow(error, 2);
	}
	return sum;
}
