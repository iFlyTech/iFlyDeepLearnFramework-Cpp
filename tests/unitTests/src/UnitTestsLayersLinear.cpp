
#include "UnitTestsLayersLinear.hpp"

using namespace DeepLearningFramework;

void Layers::UnitTestsLayersLinear::linearLayerForwardPassTest() {
  std::cout << "Forward test:" << std::endl;

  int inputFeaturesNumber = 2, outputFeaturesNumber = 4;
  Layers::Linear linearLayer(inputFeaturesNumber, outputFeaturesNumber);

  Eigen::MatrixXf weights{
      {0.5f, 0.1f, -0.5f, 0.1f},
      {0.09f, -0.5f, 0.1f, 0.09f},
  };
  Eigen::MatrixXf bias{{-0.2f, 1.f, 0.f, -0.5f}};

  Eigen::MatrixXf x{
      {-9.f, -5.f},
      {1.f, -3.f},
      {-2.f, 7.f},
  };

  Eigen::MatrixXf target{
      {-4.75f, 0.6f, 4.f, -0.85f},
      {0.43f, 0.6f, -0.8f, 0.33f},
      {-0.17f, -4.7f, 1.7f, 0.93f},
  };

  linearLayer.setWeightsAndBias(weights, bias);

  Eigen::MatrixXf out;
  linearLayer.forward(out, x);