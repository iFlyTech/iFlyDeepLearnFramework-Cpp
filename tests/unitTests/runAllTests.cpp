#include "UnitTestsActivationsReLU.hpp"
#include "UnitTestsActivationsSoftmax.hpp"
#include "UnitTestsLayersLinear.hpp"
#include "UnitTestsLossesMSE.hpp"

using namespace DeepLearningFramework;

int main() {
  std::cout << "Linear layer unit tests" << std::endl;
  Layers::UnitTestsLayersLinear::linearLayerForwardPassTest();
  Layers::UnitTestsLayersLinear::linearLayerBackwardPassTest();

  std::cout << "MSE loss unit tests" << std::endl;
  Losses::UnitTestsLossesMSE::mseLossForwardPassTest();
  Losses::UnitTestsLossesM