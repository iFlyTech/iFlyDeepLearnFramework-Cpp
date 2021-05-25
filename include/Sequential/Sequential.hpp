
/**
 * Sequential model class definition
 */

#pragma once

#include "MSE.hpp"
#include "Module.hpp"

#include <iostream>
#include <vector>

namespace DeepLearningFramework {
/**
 * Sequential class.
 *
 * Class used to store in sequence multiple modules to create a neural network
 *
 * forward: apply forward pass for each module in sequence
 * backward: calculate loss and apply backward pass for each layer in reverse
 * order.
 */
class Sequential {
public:
  Sequential(std::vector<Module *> &model, Losses::MSE loss);
  ~Sequential() {
    std::vector<Module *>::iterator it;
    for (it = mModel.begin(); it != mModel.end(); it++)
      delete (*it);
  }

  /**
   * Apply forward pass for each layer in sequence.
   *
   * @param[in/out] x data on which to apply the model (all layers in sequence).
   * Modified with neural network result
   */