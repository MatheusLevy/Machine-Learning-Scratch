#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <cmath>
#include <vector>

class Neuron {
	std::vector<double> weights;
	double preActivation;
	double activatedOutput;
	double outputDerivative;
	double error;
	double alpha;

public:
	Neuron(int, int);
	~Neuron();
	void initializeWeightes(int previousLayerSize, int currentLayerSize);
	void setError(double);
	void setWeight(double, int);
	double calculatePreActivation(std::vector<double>);
	double activate();
	double calculateOutputDerivate();
	double sigmoid();
	double relu();
	double leakyRelu();
	double inverseSqrtRelu();
	double getOutput();
	double detOutputDerivative();
	double getError();
	std::vector<double> getWeights();
};
#endif // !__NEURON_HPP
