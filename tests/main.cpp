
#include "DataBuilder2D.hpp"
#include "Linear.hpp"
#include "MSE.hpp"
#include "Metrics.hpp"
#include "ReLU.hpp"
#include "Sequential.hpp"
#include "Softmax.hpp"
#include "Trainer2D.hpp"

using namespace DeepLearningFramework;

int main() {
  /* Model creation */
  std::vector<Module *> layers;
  int inputFeaturesNumber = 2, outputFeaturesNumber = 2, hiddenSize = 10;