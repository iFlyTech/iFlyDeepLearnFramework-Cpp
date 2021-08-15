/**
 * Sequential model class implementation
 */

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

Sequential::Sequential(std::vector<Module *> &model, Losses::MSE loss) {
  mModel = model;
  mLoss = loss;
}

void Sequential::forward(Eigen::MatrixXf &x) {
  std::vector<Module *>::iterator it;
  for (it = mModel.begin(); it != mModel.end(); it++)
    (*it)->forward(x, x);
}

void Sequential::backward(float &loss, const Eigen::MatrixXf &y,
                          Eigen::MatrixXf &yPred) {
  // calculate loss
  mLoss.forward(loss, y, yPred);

  // back propagation
  Eigen::MatrixXf lossDerivative;
  mLoss.backward(lossDerivative, y, yPred);
  for (auto it = mModel.rbegin(); it != mModel.rend(); it++)
    (*it)->backward(lossDerivativ