#include "UnitTestsActivationsReLU.hpp"

using namespace DeepLearningFramework;

void Activations::UnitTestsActivationsReLU::reluActivationForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Activations::ReLU reluActivation;

  Eigen::Matrix