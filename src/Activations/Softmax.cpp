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

void Softmax::backward(Eigen::MatrixXf &ddout, const Eigen::MatrixXf &dout) {
  const Eigen::MatrixXf grad = dout;

  for (int i = 0; i < dout.rows(); ++i) {
    for (int j = 0; j < dout.cols(); ++j) {
      for (int k = 0; k < dout.cols(); ++k) {
        if (j == k) {
          ddout(i, j) += grad(i, k) *
                         mForwardInputWithSoftmaxApplied(i, k) *
                         (1.f - mForwardInputWithSoftmaxApplied(i, j