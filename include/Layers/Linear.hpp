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
  Linear(int inputFeaturesNumber, int outputFeaturesNumber);
  ~Linear() = default;

  /**
   * Forward pass of the Linear layer.
   *
   * @param[out] out input * weights + bias
   * @param[in] x Values on which to apply weights and biases.
   */
  void forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) override;

  /**
   * Backward pass of the Linear layer.
   *
   * @param[out] ddout input * weights
   * @param[in] dout Values on which to apply weights and biases.
   */
  void backward(Eigen::MatrixXf &ddout, const Eigen::MatrixXf &dout) override;

  /* Print description of Linear layer class */
  void printDescription() override;

  /**
   * Set learning rate used to update weights and bias.
   *
   * @param[in] lr learning rate to use.
   */
  void setLR(float lr);

  /** Get the number of parameters of the Linear layer. */
  uint32_t getParametersCount();

  /** get weights */
  Eigen::MatrixXf getWeights();

  /** get bias */
  Eigen::MatrixXf getBias();

  /** set weights and bias for unit testings purpose */
  void setWeightsAndBi