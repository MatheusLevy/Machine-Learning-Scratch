#ifndef __HIDDEN_LAYER_HPP
#define __HIDDEN_LAYER_HPP

#include "layer.hpp"

class HiddenLayer : public Layer {
public:
	HiddenLayer(int prev, int curr) : Layer(prev, curr){}
	void backProp(Layer next);
	void updateWeights(double, Layer *);
};
#endif // !__HIDDEN_LAYER_HPP
