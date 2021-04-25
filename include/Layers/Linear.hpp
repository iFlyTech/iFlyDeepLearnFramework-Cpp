/**
 * Linear layer class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Layers {
/**
 * Layer class: Linear.
 *
 * forward: output = input * weights + bias
 * backwar