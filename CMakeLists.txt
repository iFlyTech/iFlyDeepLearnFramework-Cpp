cmake_minimum_required(VERSION 3.0.0)
project(TestDeepLearningFramework VERSION 0.1.0)

include(CTest)
enable_testing()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/Activations
    ${PROJECT_SOURCE_DIR}/include/Data
    ${PROJECT_SOURCE_DIR}/include/Layer