/**
 * Softmax activation class implementation
 */

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax() {}

void Softmax::forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) {
  Softmax::equation(out, x);
  mForwardInputWithSoftmaxApplied = out;
}

void Softmax::backward(Eigen::MatrixXf &ddout, const Eigen::MatrixXf &dou