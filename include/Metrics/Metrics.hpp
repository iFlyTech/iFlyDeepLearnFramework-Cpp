/**
 * Metrics class definition
 */

#pragma once

#include <Eigen/Dense>

namespace DeepLearningFramework {
/**
 * Metrics class
 *
 * accuracy: count of good predictions / number of predictions
 */
class Metrics {
public:
  Metrics() = delete;
  ~Metrics() = delete;

  /**
   * accuracy static method
   *
   * accuracy: count of good predictions / number of predictio