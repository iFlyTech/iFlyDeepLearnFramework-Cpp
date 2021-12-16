#include "UnitTestsLossesMSE.hpp"

using namespace DeepLearningFramework;

void Losses::UnitTestsLossesMSE::mseLossForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Losses::MSE mseLoss;

  Eigen::MatrixXf y{
      {1.f, 0.f},
      {1.f, 0.f},
      {0.f, 1.f},
  };

  Eigen::MatrixXf yPred{
      {0.4f, 0.6f},
      {0.2f, 0.8f},
      {0.9f, 0.1f},
  };

  float target = 1.20667f;

  float out;
  