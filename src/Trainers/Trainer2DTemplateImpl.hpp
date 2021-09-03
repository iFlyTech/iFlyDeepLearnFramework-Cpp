/**
 * Template function from Trainer2D class implementation
 */

using namespace DeepLearningFramework;

template <uint32_t batchSize>
void Trainer2D::trainModel(
    std::vector<float> trainLossHistory,
    std::vector<float> trainAccuracyHistory, std::vector<float> testLossHistory,
    std::vector<float> testAccuracyHistory, Sequential &model,
    uin