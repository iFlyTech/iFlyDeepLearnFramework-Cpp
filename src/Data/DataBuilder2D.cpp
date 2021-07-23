/**
 * Build Data class implementation
 */

#include "DataBuilder2D.hpp"

using namespace DeepLearningFramework;

void DataBuilder2D::generateDiscSet(Eigen::MatrixXf &labels,
                                    Eigen::MatrixXf &features,
                                    uint32_t samplesCount, float discRadius) {
  features = Eigen::MatrixXf::Random(samplesCount, 2);
  labels = Eigen::MatrixXf(samp