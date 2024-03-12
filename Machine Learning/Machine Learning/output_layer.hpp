#ifndef __OUTPUT_LAYER_HPP
#define __OUTPUT_LAYER_HPP

#include "layer.hpp"
#include "data.hpp"

class OutputLayer : public Layer {
public:
	OutputLayer(int prev, int curr) : Layer(prev, curr){}
	void feedForward(Layer);
	void backProp(Data* data);
	void updateWeights(double, Layer*);
};
#endif // !__OUTPUT_LAYER_HPO
