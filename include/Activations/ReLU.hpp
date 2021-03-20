/**
 * ReLU activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: ReLU.
 *
 * forward: output = input if input > 0, else 0
 * backward: output = 1*input if forward input was > 0, else 0
 */
class ReLU : public Module {
public:
  ReLU();
  ~ReLU() = de