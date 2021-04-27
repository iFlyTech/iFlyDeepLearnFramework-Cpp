/**
 * Linear layer class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Layers {
/**
 * Layer class: Linear.
 *
 * forward: output = input * weights + bias
 * backward: update Weights nd Bias; output = input * weights
 */
class Linear : public Module {
public:
  Linear(int inputFeaturesNumber, int outputFe