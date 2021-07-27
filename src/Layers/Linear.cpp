
/**
 * Linear layer class implementation
 */

#include "Linear.hpp"

#include <iostream>

using namespace DeepLearningFramework::Layers;

Linear::Linear(int inputFeaturesNumber, int outputFeaturesNumber) {
  mInputFeaturesNumber = inputFeaturesNumber;
  mOutputFeaturesNumber = outputFeaturesNumber;
  mWeights = Eigen::MatrixXf::Random(inputFeaturesNumber, outputFeaturesNumber);
  mBias = Eigen::MatrixXf::Random(1, outputFeaturesNumber);
}

void Linear::forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) {
  mForwardInput = x;
  out = x.matrix() * mWeights.matrix();
  for (int row = 0; row < out.rows(); ++row) {
    out.row(row).array() -= mBias.array();
  }
}

void Linear::backward(Eigen::MatrixXf &ddout, const Eigen::MatrixXf &dout) {
  // update weights and bias
  mWeights =
      mWeights.array() - mLR * (mForwardInput.transpose() * dout).array();
  mBias = mBias.array() - mLR * dout.colwise().mean().array();

  // calculate output
  ddout = dout * mWeights.transpose();
}

void Linear::printDescription() {
  std::cout << "Linear Layer [" << mInputFeaturesNumber << ", "
            << mOutputFeaturesNumber << "], "
            << "parameters: " << this->getParametersCount()
            << ", learning rate: " << mLR << std::endl;
}