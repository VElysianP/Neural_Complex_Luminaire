// #pragma once

#include <TensorRT/NvInfer.h>
#include <TensorRT/NvOnnxParser.h>
#include <onnx-tensorrt/cudaWrapper.h>
#include <onnx-tensorrt/ioHelper.h>

#include <NvCaffeParser.h>
#include <NvUtils.h>
#include <NvInfer.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>

using namespace cudawrapper;

static nvinfer1::Logger gLogger;

// Number of times we run inference to calculate average time.
constexpr int ITERATIONS = 10;
// Maxmimum absolute tolerance for output tensor comparison against reference.
constexpr double ABS_EPSILON = 0.005;
// Maxmimum relative tolerance for output tensor comparison against reference.
constexpr double REL_EPSILON = 0.05;
// Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 31; // 1 GB

static int getBindingInputIndex(nvinfer1::IExecutionContext* context);

// NOTE(yaoyi): for eval network 
nvinfer1::ICudaEngine* lf_createCudaEngine(std::string const& onnxModelPath, int batchSize);
void lf_launchInference(nvinfer1::IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize);


// NOTE(yaoyi): for sampling weight network
nvinfer1::ICudaEngine* sw_createCudaEngine(std::string const& onnxModelPath, int batchSize);
void sw_launchInference(nvinfer1::IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize);

// NOTE(yaoyi): for pdf network
// nvinfer1::ICudaEngine* pdf_createCudaEngine(std::string const& onnxModelPath, int batchSize);
// void pdf_launchInference(nvinfer1::IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize);