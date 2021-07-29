/**
 * MSE loss class implementation
 */

#include "MSE.hpp"

#include <iostream>

using namespace DeepLearningFramework::Losses;

MSE::MSE() {}

void MSE::forward(float &loss, const Eig