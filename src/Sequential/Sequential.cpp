/**
 * Sequential model class implementation
 */

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

Sequential::Sequential(std::vector<Module *> &model, Losses::MSE loss) {
  mModel = model;
  m