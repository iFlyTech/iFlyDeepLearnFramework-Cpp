/**
 * ReLU activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: ReLU.
 *
 * forward: output = input if input > 0, else 0
 * backward: outp