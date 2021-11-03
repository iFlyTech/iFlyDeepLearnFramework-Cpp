#include "UnitTestsActivationsReLU.hpp"

using namespace DeepLearningFramework;

void Activations::UnitTestsActivationsReLU::reluActivationForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Activations::ReLU reluActivation;

  Eigen::MatrixXf x{
      {-9.f, -5.f, 0.f, 1.f, 2.f, 8.f},
      {9.f, -5.f, 0.f, 1.f, -2.f, 8.f},
      {-4.f, 5.f, 0.f, 1.f, 2.f, -8.f},
      {-2.f, -5.f, 0.f, 1.f, -2.f, 8.f},
  };

  Eigen::MatrixXf target{
      {0.f, 0.f, 0.f, 1.f, 2.f, 8.f},
      {9.f, 0.f, 0.f, 1.f, 0.f, 8.f},
      {0.f, 5.f, 0.f, 1.f, 2.f, 0.f},
      {0.f, 0.f, 0.f, 1.f, 0.f, 8.f},
  };

  Eigen::MatrixXf out;
  reluActivation.forward(out, x);

  if (!target.isApprox(out)) {
    std::cout << "Result KO" << std::endl;
    std::cout << "Expect: " << target << std::endl;
    std::cout << "Got: " << out << std::endl;
    return;
  }

  std::cout << "OK" << std::endl;
}

void Activations::UnitTestsActivationsReLU::reluActivationBackwardPassTest() {
  std::cout << "Backward test:" << std::endl;

  Activations::ReLU reluActivation;

  // forward in