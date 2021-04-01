/**
 * Softmax activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: Softmax.
 *
 * forward: output = exp(IN_i)/exp(sum(IN)), input saved for backward p