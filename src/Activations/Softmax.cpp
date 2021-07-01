/**
 * Softmax activation class implementation
 */

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax() {}

void Softmax::forward(Eigen::MatrixXf &out, cons