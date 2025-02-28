
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
  layers.emplace_back(
      new Layers::Linear((int)inputFeaturesNumber, (int)hiddenSize));
  layers.emplace_back(new Activations::ReLU());
  layers.emplace_back(new Layers::Linear((int)hiddenSize, (int)hiddenSize));
  layers.emplace_back(new Activations::ReLU());
  layers.emplace_back(
      new Layers::Linear((int)hiddenSize, (int)outputFeaturesNumber));
  layers.emplace_back(new Activations::Softmax());

  Losses::MSE mseLoss;

  Sequential model(layers, mseLoss);
  model.printDescription();

  /* Train params */
  float learningRate = 0.03f;
  // number of train and test samples
  uint32_t samplesCount = 2000;
  std::vector<float> trainLossHistory, trainAccuracyHistory, testLossHistory,
      testAccuracyHistory;
  uint32_t epochsCount = 100, verboseFrequence = 1;
  constexpr auto batchSize = 64;

  // Update learning rate for model
  model.setLR(learningRate);

  /* Generate train and test sets */
  Eigen::MatrixXf trainTarget, trainFeatures;
  DataBuilder2D::generateDiscSet(trainTarget, trainFeatures, samplesCount, 0.3);
  Eigen::MatrixXf testTarget, testFeatures;
  DataBuilder2D::generateDiscSet(testTarget, testFeatures, samplesCount, 0.3);

  // Train model
  Trainer2D::trainModel<batchSize>(trainLossHistory, trainAccuracyHistory,
                                   testLossHistory, testAccuracyHistory, model,
                                   epochsCount, trainTarget, trainFeatures,
                                   testTarget, testFeatures, verboseFrequence);
}