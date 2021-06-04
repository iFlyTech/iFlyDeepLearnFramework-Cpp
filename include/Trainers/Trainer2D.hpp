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
   * Train a model for n epoch on specified data
   *
   * @param[out] trainLossHistory loss from epoch 0 to epochsCount on train set
   * @param[out] trainAccuracyHistory accuracy from epoch 0 to epochsCount on
   * train set
   * @param[out] testLossHistory loss from epoch 0 to epochsCount on test set
   * @param[out] testAccuracyHistory accuracy from epoch 0 to epochsCount on
   * test set
   * @param[in/out] model to train
   * @param[in] e