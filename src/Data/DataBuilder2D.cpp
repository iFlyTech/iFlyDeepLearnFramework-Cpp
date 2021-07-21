/**
 * Build Data class implementation
 */

#include "DataBuilder2D.hpp"

using namespace DeepLearningFramework;

void DataBuilder2D::generateDiscSet(Eigen::MatrixXf &labels,
                                    Eigen::MatrixXf &features,
                                    uint32_t samplesCount,