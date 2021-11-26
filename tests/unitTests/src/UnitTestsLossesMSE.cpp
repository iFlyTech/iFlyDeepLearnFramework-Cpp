#include "UnitTestsLossesMSE.hpp"

using namespace DeepLearningFramework;

void Losses::UnitTestsLossesMSE::mseLossForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  Losses::MSE mseLoss;

  Eigen::MatrixXf