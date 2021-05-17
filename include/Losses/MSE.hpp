
/**
 * MSE loss class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Losses {
/**
 * Loss class: MSE.
 *
 * forward: output = input if input > 0, else 0
 * backward: output = 1*input if forward input was > 0, else 0
 */
class MSE {
public:
  MSE();
  ~MSE() = default;

  /**
   * Forward pass of the MSE loss function.