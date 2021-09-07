/**
 * Template function from Trainer2D class implementation
 */

using namespace DeepLearningFramework;

template <uint32_t batchSize>
void Trainer2D::trainModel(
    std::vector<float> trainLossHistory,
    std::vector<float> trainAccuracyHistory, std::vector<float> testLossHistory,
    std::vector<float> testAccuracyHistory, Sequential &model,
    uint32_t epochsCount, const Eigen::MatrixXf &trainTarget,
    const Eigen::MatrixXf &trainFeatures, const Eigen::MatrixXf &testTarget,
    const Eigen::MatrixXf &testFeatures, uint32_t verboseFrequence) {
  addAccuracy(trainAccuracyHistory, model, trainTarget, trainFeatures);
  addAccuracy(testAccuracyHistory, model, testTarget, testFeatures);
  uint3