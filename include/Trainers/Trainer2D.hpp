/**
 * Trainer class definition
 */

#pragma once

#include "Metrics.hpp"
#include "Sequential.hpp"

namespace DeepLearningFramework {
/**
 * Trainer class
 *
 * trainModel: train a model
 */
class Trainer2D {
public:
  Trainer2D() = delete;
  ~Trainer2D() = delete;

  /**
   * trainModel static method
   *
   * Train a model for n epoch 