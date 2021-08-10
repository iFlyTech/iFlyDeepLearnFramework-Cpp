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
    (*