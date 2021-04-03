/**
 * Softmax activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: Softmax.
 *
 * forward: output = exp(IN_i)/exp(sum(IN)), input saved for backward pass
 * backward: output = [Softmax(forward_input) * (1 - Softmax(forward_input))] *
 * input
 */
class Softmax : public Module {
public:
  Softmax();
  ~Softmax() = default;

  /**
   * Forward pass of the Soft