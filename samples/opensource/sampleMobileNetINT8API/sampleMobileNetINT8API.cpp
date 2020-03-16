/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! sampleINT8API_mod.cpp
//! Modified to run pre-quantized MobileNetV1 for classification from MLPerf Inference 0.5 suite.
//! This file contains implementation showcasing usage of INT8 calibration and precision APIs.
//! It creates classification networks such as mobilenet, vgg19, resnet-50 from onnx model file.
//! This sample showcases setting per tensor dynamic range overriding calibrator generated scales if it exists.
//! This sample showcases how to set computation precision of layer. It involves forcing output tensor type of the layer
//! to particular precision. It can be run with the following command line: Command: ./sample_int8_api_mod [-h or --help]
//! [-m modelfile] [-s per_tensor_dynamic_range_file] [-i image_file] [-r reference_file] [-d path/to/data/dir]
//! [--verbose] [--useDLACore <id>]

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

const std::string gSampleName = "TensorRT.sample_mobilenet_int8_api";

struct SampleINT8APIPreprocessing
{
    // Based on: https://github.com/mlperf/inference_results_v0.5/blob/master/closed/NVIDIA/scripts/preprocess_data.py#L131
    std::vector<char> mean{128, 128, 128};
    // std and scale are not needed for INT8 inputs here.
    std::vector<int> inputDims{3, 224, 224};
};

//!
//! \brief The SampleINT8APIParams structure groups the additional parameters required by
//!         the INT8 API sample
//!
struct SampleINT8APIParams
{
    bool verbose{false};
    bool writeNetworkTensors{false};
    int dlaCore{-1};
    int batchSize{1};
    int topBottomK{10};
    bool fp32{false};
    bool safeGpuInt8{false};

    SampleINT8APIPreprocessing mPreproc;
    std::string modelFileName;
    std::vector<std::string> dataDirs;
    std::string dynamicRangeFileName;
    std::string imageFileName;
    std::string referenceFileName;
    std::string networkTensorsFileName;
};

//!
//! \brief The SampleINT8API class implements INT8 inference on classification networks.
//!
//! \details INT8 API usage for setting custom INT8 range for each input layer. API showcase how
//!           to perform INT8 inference without calibration table
//!
class SampleINT8API
{
private:
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleINT8API(const SampleINT8APIParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    Logger::TestResult build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    Logger::TestResult infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    Logger::TestResult teardown();

    SampleINT8APIParams mParams; //!< Stores Sample Parameter

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> mInOut; //!< Input and output mapping of the network

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network

    std::unordered_map<std::string, float>
        mPerTensorDynamicRangeMap; //!< Mapping from tensor name to max absolute dynamic range values

    void getInputOutputNames(); //!< Populates input and output mapping of the network

    //!
    //! \brief Reads the ppm input image, preprocesses, and stores the result in a managed buffer
    //!
    template <typename input_precision_t> bool prepareInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Modified samplesCommon::classify with option to print output activation values
    //!
    template <typename result_vector_t> inline std::vector<std::string> classify_verbose(
        const std::vector<std::string>& refVector, const result_vector_t& output, const size_t topK) const;

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers) const;

    //!
    //! \brief Populate per tensor dynamic range values
    //!
    bool readPerTensorDynamicRangeValues();

    //!
    //! \brief  Sets custom dynamic range for network tensors
    //!
    bool setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief  Write network tensor names to a file.
    //!
    void writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
};

//!
//! \brief  Populates input and output mapping of the network
//!
void SampleINT8API::getInputOutputNames()
{
    int nbindings = mEngine.get()->getNbBindings();
    assert(nbindings == 2);
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        if (mEngine.get()->bindingIsInput(b))
        {
            if (mParams.verbose)
            {
                gLogInfo << "Found input: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            mInOut["input"] = mEngine.get()->getBindingName(b);
        }
        else
        {
            if (mParams.verbose)
            {
                gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            mInOut["output"] = mEngine.get()->getBindingName(b);
        }
    }
}

//!
//! \brief Populate per tensor dynamic range values
//!
bool SampleINT8API::readPerTensorDynamicRangeValues()
{
    std::ifstream iDynamicRangeStream(mParams.dynamicRangeFileName);
    if (!iDynamicRangeStream)
    {
        gLogError << "Could not find per tensor scales file: " << mParams.dynamicRangeFileName << std::endl;
        return false;
    }

    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line))
    {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        float dynamicRange = std::stof(token);
        mPerTensorDynamicRangeMap[tensorName] = dynamicRange;
    }
    return true;
}

//!
//! \brief  Write network tensor names to a file.
//!
void SampleINT8API::writeNetworkTensorNames(const SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    gLogInfo << "Sample requires to run with per tensor dynamic range." << std::endl;
    gLogInfo << "In order to run INT8 inference without calibration, user will need to provide dynamic range for all "
                "the network tensors." << std::endl;

    std::ofstream tensorsFile{mParams.networkTensorsFileName};

    // Iterate through network inputs to write names of input tensors.
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string tName = network->getInput(i)->getName();
        tensorsFile << "TensorName: " << tName << std::endl;
        if (mParams.verbose)
        {
            gLogInfo << "TensorName: " << tName << std::endl;
        }
    }

    // Iterate through network layers.
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        // Write output tensors of a layer to the file.
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            string tName = network->getLayer(i)->getOutput(j)->getName();
            tensorsFile << "TensorName: " << tName << std::endl;
            if (mParams.verbose)
            {
                gLogInfo << "TensorName: " << tName << std::endl;
            }
        }
    }
    tensorsFile.close();
    gLogInfo << "Successfully generated network tensor names. Writing: " << mParams.networkTensorsFileName << std::endl;
    gLogInfo << "Use the generated tensor names file to create dynamic range file for INT8 inference. Follow README.md "
                "for instructions to generate dynamic_ranges.txt file." << std::endl;
}

//!
//! \brief  Sets custom dynamic range for network tensors
//!
bool SampleINT8API::setDynamicRange(SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // populate per tensor dynamic range
    if (!readPerTensorDynamicRangeValues())
    {
        return false;
    }

    gLogInfo << "Setting Per Tensor Dynamic Range" << std::endl;
    if (mParams.verbose)
    {
        gLogInfo << "If dynamic range for a tensor is missing, TensorRT will run inference assuming dynamic range for "
                    "the tensor as optional."
                 << std::endl;
        gLogInfo << "If dynamic range for a tensor is required then inference will fail. Follow README.md to generate "
                    "missing per tensor dynamic range."
                 << std::endl;
    }

    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string tName = network->getInput(i)->getName();
        // Set input type to INT8:
        network->getInput(i)->setType(DataType::kINT8);
        float scaleValue = 127.0f;
        if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
        {
            scaleValue = mPerTensorDynamicRangeMap.at(tName);
        }
        else
        {
            if (mParams.verbose)
            {
                gLogWarning << "Missing dynamic range for input tensor: " << tName << ", "
                               "using default scales of 127." << std::endl;
            }
        }
        network->getInput(i)->setDynamicRange(-scaleValue, scaleValue);
    }

    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j)
        {
            string tName = network->getLayer(i)->getOutput(j)->getName();
            float scaleValue = 127.0f;
            if (mPerTensorDynamicRangeMap.find(tName) != mPerTensorDynamicRangeMap.end())
            {
                // Calibrator generated dynamic range for network tensor can be overridden or set using below API
                scaleValue = mPerTensorDynamicRangeMap.at(tName);
            }
            else
            {
                if (mParams.verbose)
                {
                    gLogWarning << "Missing dynamic range for tensor: " << tName << ", "
                                   "using default scales of 127." << std::endl;
                }
            }
            network->getLayer(i)->getOutput(j)->setDynamicRange(-scaleValue, scaleValue);
        }
    }

    if (mParams.verbose)
    {
        gLogInfo << "Per Tensor Dynamic Range Values for the Network (read from the scales text file):" << std::endl;
        for (auto iter = mPerTensorDynamicRangeMap.begin(); iter != mPerTensorDynamicRangeMap.end(); ++iter)
        {
            gLogInfo << "Tensor: " << iter->first << ". Max Absolute Dynamic Range: " << iter->second
                     << " (resolution at INT8 after tensor quantization: " << (iter->second / 127.0f) << ")" << std::endl;
        }
    }
    return true;
}

//!
//! \brief Preprocess inputs and allocate host/device input buffers
//!
template <typename input_precision_t> bool SampleINT8API::prepareInput(const samplesCommon::BufferManager& buffers)
{
    if (samplesCommon::toLower(samplesCommon::getFileType(mParams.imageFileName)).compare("ppm") != 0)
    {
        gLogError << "Wrong format: " << mParams.imageFileName << " is not a ppm file." << std::endl;
        return false;
    }

    int channels = mParams.mPreproc.inputDims.at(0);
    int height = mParams.mPreproc.inputDims.at(1);
    int width = mParams.mPreproc.inputDims.at(2);
    int max{0};
    std::string magic{""};

    vector<uint8_t> fileData(channels * height * width);
    // Prepare PPM Buffer to read the input image
    // samplesCommon::PPM<channels, height, width> ppm;
    // samplesCommon::readPPMFile(mParams.imageFileName, ppm);

    std::ifstream infile(mParams.imageFileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(fileData.data()), width * height * channels);

    input_precision_t* hostInputBuffer = static_cast<input_precision_t*>(buffers.getHostBuffer(mInOut["input"]));

    // Convert HWC to CHW and Normalize
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = c * height * width + h * width + w;
                int srcIdx = h * width * channels + w * channels + c;
                // This equation includes 2 steps:
                // 1. Normalize Image using per-channel mean (here: uniformly 128).
                // 2. Shuffle HWC to CHW form
                // See https://github.com/mlperf/inference_results_v0.5/blob/master/closed/NVIDIA/scripts/preprocess_data.py#L131 for reference.
                hostInputBuffer[dstIdx] = static_cast<input_precision_t>(fileData[srcIdx] - mParams.mPreproc.mean.at(c));
            }
        }
    }
    return true;
}

//!
//! \brief Modified samplesCommon::classify with option to print output activation values
//!
template <typename result_vector_t>
inline std::vector<std::string> SampleINT8API::classify_verbose(
    const std::vector<std::string>& refVector, const result_vector_t& output, const size_t topBottomK) const
{
    auto indsTop = samplesCommon::argsort(output.cbegin(), output.cend(), true);
    auto indsBottom = samplesCommon::argsort(output.cbegin(), output.cend(), false);
    std::vector<std::string> result;
    for (size_t k = 0; k < topBottomK; ++k)
    {
        const size_t index = indsTop[k];
        result.push_back(refVector[index]);
        // if (mParams.verbose)
        // {
        gLogInfo << "Top-" << (k + 1) << " predicted class, activation value: " << refVector[index] << ", "
                    << output[index] << std::endl;
        // }
    }
    // if (mParams.verbose)
    // {
    for (size_t k = 0; k < topBottomK; ++k)
    {
        const size_t index = indsBottom[k];
        gLogInfo << "Bottom-" << (k + 1) << " predicted class, activation value: " << refVector[index] << ", "
                    << output[index] << std::endl;
    }
    // }
    return result;
}

//!
//! \brief Verifies that the output is correct and prints it
//!
// template <typename output_precision_t>
bool SampleINT8API::verifyOutput(const samplesCommon::BufferManager& buffers) const
{
    // copy output host buffer data for further processing
    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    vector<float> output(probPtr, probPtr + mOutputDims.d[0] * mParams.batchSize);

    auto inds = samplesCommon::argsort(output.cbegin(), output.cend(), true);

    // read reference labels to generate prediction labels
    vector<string> referenceVector;
    if (!samplesCommon::readReferenceFile(mParams.referenceFileName, referenceVector))
    {
        gLogError << "Unable to read reference file: " << mParams.referenceFileName << std::endl;
        return false;
    }
    // vector<string> top5Result = samplesCommon::classify(referenceVector, output, 5);
    vector<string> topKResult = classify_verbose(referenceVector, output, mParams.topBottomK);

    gLogInfo << "SampleINT8API result - Detected:" << std::endl;
    for (int i = 1; i <= mParams.topBottomK; ++i)
    {
        gLogInfo << "[" << i << "]  " << topKResult[i - 1] << std::endl;
    }

    return true;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates INT8 classification network by parsing the onnx model and builds
//!          the engine that will be used to run INT8 inference (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
Logger::TestResult SampleINT8API::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        gLogError << "Unable to create builder object." << std::endl;
        return Logger::TestResult::kFAILED;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        gLogError << "Unable to create network object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        gLogError << "Unable to create config object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        gLogError << "Unable to create parser object." << mParams.referenceFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;
    if (!parser->parseFromFile(mParams.modelFileName.c_str(), verbosity))
    {
        gLogError << "Unable to parse ONNX model file: " << mParams.modelFileName << std::endl;
        return Logger::TestResult::kFAILED;
    }

    if (mParams.writeNetworkTensors)
    {
        writeNetworkTensorNames(network);
        return Logger::TestResult::kWAIVED;
    }



    // Configure builder
    auto maxBatchSize = mParams.batchSize;
    config->setMaxWorkspaceSize(1_GiB);

    if (!mParams.fp32) // equivalent to INT8 mode (default)
    {
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
        if (!builder->platformHasFastInt8())
        {
            gLogError << "Platform does not support INT8 inference. sampleINT8API can only run in INT8 Mode if --fp32 is not passed." << std::endl;
            return Logger::TestResult::kWAIVED;
        }
        // Enable INT8 model. Required to set custom per tensor dynamic range or INT8 Calibration
        config->setFlag(BuilderFlag::kINT8);
        // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
        config->setInt8Calibrator(nullptr);

        if(mParams.dlaCore >= 0)
        {
            samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
            if (maxBatchSize > builder->getMaxDLABatchSize())
            {
                std::cerr << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                          << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
                maxBatchSize = builder->getMaxDLABatchSize();
            }
        }
        else if (mParams.safeGpuInt8)
        {
            builder->setEngineCapability(nvinfer1::EngineCapability::kSAFE_GPU);
        }

        if (!setDynamicRange(network))
        {
            gLogError << "Unable to set per-tensor dynamic range." << std::endl;
            return Logger::TestResult::kFAILED;
        }
    }
    builder->setMaxBatchSize(maxBatchSize);

    // build TRT engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        gLogError << "Unable to build CUDA engine." << std::endl;
        return Logger::TestResult::kFAILED;
    }

    // populates input output map structure
    getInputOutputNames();

    // derive input/output dims from engine bindings
    const int inputIndex = mEngine.get()->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine.get()->getBindingDimensions(inputIndex);

    const int outputIndex = mEngine.get()->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine.get()->getBindingDimensions(outputIndex);

    return Logger::TestResult::kRUNNING;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output
//!
Logger::TestResult SampleINT8API::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return Logger::TestResult::kFAILED;
    }
    // Read the input data into the managed buffers
    // There should be just 1 input tensor
    bool prepareInputSuccess = false;
    if (mParams.fp32)
    {
        prepareInputSuccess = prepareInput<float>(buffers);
    }
    else
    {
        prepareInputSuccess = prepareInput<char>(buffers);
    }
    if (!prepareInputSuccess)
    {
        return Logger::TestResult::kFAILED;
    }

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return Logger::TestResult::kFAILED;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    // Check and print the output of the inference
    return verifyOutput(buffers) ? Logger::TestResult::kRUNNING : Logger::TestResult::kFAILED;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
Logger::TestResult SampleINT8API::teardown()
{
    return Logger::TestResult::kRUNNING;
}

//!
//! \brief The SampleINT8APIArgs structures groups the additional arguments required by
//!         the INT8 API sample
//!
struct SampleINT8APIArgs : public samplesCommon::Args
{
    bool verbose{false};
    bool writeNetworkTensors{false};
    std::string modelFileName{"mobilenet_quantized_opt.onnx"};
    std::string imageFileName{"airliner.ppm"};
    std::string referenceFileName{"reference_labels.txt"};
    std::string dynamicRangeFileName{"mobilenet_last_dynamic_range.txt"};
    std::string networkTensorsFileName{"network_tensors.txt"};
    int topBottomK{5};
    bool runInFp32{false};
    bool safeGpuInt8{false};
};

//! \brief This function parses arguments specific to SampleINT8API
//!
bool parseSampleINT8APIArgs(SampleINT8APIArgs& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--model=", 8))
        {
            args.modelFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--image=", 8))
        {
            args.imageFileName = (argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--reference=", 12))
        {
            args.referenceFileName = (argv[i] + 12);
        }
        else if (!strncmp(argv[i], "--write_tensors", 15))
        {
            args.writeNetworkTensors = true;
        }
        else if (!strncmp(argv[i], "--network_tensors_file=", 23))
        {
            args.networkTensorsFileName = (argv[i] + 23);
        }
        else if (!strncmp(argv[i], "--ranges=", 9))
        {
            args.dynamicRangeFileName = (argv[i] + 9);
        }
        else if (!strncmp(argv[i], "--fp32", 6))
        {
            args.runInFp32 = true;
        }
        else if (!strncmp(argv[i], "--useDLACore=", 13))
        {
            args.useDLACore = std::stoi(argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--topBottomK=", 13))
        {
            args.topBottomK = std::stoi(argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--safeGpuInt8", 9))
        {
            args.safeGpuInt8 = true;
        }
        else if (!strncmp(argv[i], "--data=", 7))
        {
            std::string dirPath = (argv[i] + 7);
            if (dirPath.back() != '/')
            {
                dirPath.push_back('/');
            }
            args.dataDirs.push_back(dirPath);
        }
        else if (!strncmp(argv[i], "--verbose", 9) || !strncmp(argv[i], "-v", 2))
        {
            args.verbose = true;
        }
        else if (!strncmp(argv[i], "--help", 6) || !strncmp(argv[i], "-h", 2))
        {
            args.help = true;
        }
        else
        {
            gLogError << "Invalid Argument: " << argv[i] << std::endl;
            return false;
        }
    }
    if ((args.useDLACore >= 0) && (args.runInFp32))
    {
        gLogError << "Cannot set --useDLACore=N (where N>=0) at the same time as --fp32. "
                     "Exiting." << std::endl;
        return false;
    }

    if ((args.safeGpuInt8) && ((args.runInFp32) || (args.useDLACore >= 0)))
    {
        gLogError << "Tried to set --safeGpuInt8 with --useDLACore=N (where N>=0) or --fp32. "
                     "For safe DLA inference, please save DLA loadable, "
                     "then use e.g. dla_safety_runtime to run inference with saved DLA loadable, "
                     "or alternatively run with your own application" << std::endl;
        return false;
    }

    return true;
}

void validateInputParams(SampleINT8APIParams& params)
{
    gLogInfo << "Please follow README.md to generate missing input files." << std::endl;
    gLogInfo << "Validating input parameters. Using following input files for inference." << std::endl;
    params.modelFileName = locateFile(params.modelFileName, params.dataDirs);
    gLogInfo << "    Model File: " << params.modelFileName << std::endl;
    if (params.writeNetworkTensors)
    {
        gLogInfo << "    Writing Network Tensors File to: " << params.networkTensorsFileName << std::endl;
        return;
    }
    params.imageFileName = locateFile(params.imageFileName, params.dataDirs);
    gLogInfo << "    Image File: " << params.imageFileName << std::endl;
    params.referenceFileName = locateFile(params.referenceFileName, params.dataDirs);
    gLogInfo << "    Reference File: " << params.referenceFileName << std::endl;
    params.dynamicRangeFileName = locateFile(params.dynamicRangeFileName, params.dataDirs);
    gLogInfo << "    Dynamic Range File: " << params.dynamicRangeFileName << std::endl;
    return;
}

//!
//! \brief This function initializes members of the params struct using the command line args
//!
SampleINT8APIParams initializeSampleParams(SampleINT8APIArgs args)
{
    SampleINT8APIParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/samples/int8_api/");
        params.dataDirs.push_back("data/int8_api/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.dataDirs.push_back(""); //! In case of absolute path search
    params.batchSize = 1;
    params.verbose = args.verbose;
    params.modelFileName = args.modelFileName;
    params.imageFileName = args.imageFileName;
    params.referenceFileName = args.referenceFileName;
    params.dynamicRangeFileName = args.dynamicRangeFileName;
    params.dlaCore = args.useDLACore;
    params.writeNetworkTensors = args.writeNetworkTensors;
    params.networkTensorsFileName = args.networkTensorsFileName;
    params.topBottomK = args.topBottomK;
    params.fp32 = args.runInFp32;
    params.safeGpuInt8 = args.safeGpuInt8;
    validateInputParams(params);
    return params;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout <<    "Usage: ./sample_mobilenet_int8_api [-h or --help] [--model=model_file] [--ranges=per_tensor_dynamic_range_file]\n"
                    "[--image=image_file] [--reference=reference_file] [--write_tensors] [--network_tensors_file=network_tensors_file]\n"
                    "[--data=/path/to/data/dir] [--useDLACore=<int>] [--topBottomK=<int>] [--fp32] [--safeGpuInt8]\n"
                    "[-v or --verbose]"
                    << std::endl;
    std::cout <<    "-h or --help. Display This help information"<< std::endl;
    std::cout <<    "--model=model_file.onnx or /absolute/path/to/model_file.onnx. Generate model file using README.md in case\n"
                    "it does not exists. Defaults to mobilenet_quantized_opt.onnx."
                    << std::endl;

    std::cout <<    "--image=image.ppm or /absolute/path/to/image.ppm. Image to infer. Defaults to airliner.ppm."
                    << std::endl;
    std::cout <<    "--reference=reference.txt or /absolute/path/to/reference.txt. Reference labels file. Defaults to\n"
                    "reference_labels.txt."
                    << std::endl;
    std::cout <<    "--ranges=ranges.txt or /absolute/path/to/ranges.txt. Specify custom per tensor dynamic range for the\n"
                    "network. Defaults to mobilenet_last_dynamic_range.txt."
                    << std::endl;
    std::cout <<    "--write_tensors. Option to generate file containing network tensors name. By default writes network_tensors.txt.\n"
                    "To provide user defined file name use additional option --network_tensors_file. See --network_tensors_file option\n"
                    "usage for more detail."
                    << std::endl;
    std::cout <<    "--network_tensors_file=network_tensors.txt or /absolute/path/to/network_tensors.txt. This option\n"
                     "needs to be used with --write_tensors option. Specify file name (will write to current execution\n"
                     "directory) or absolute path to file name to write network tensor names file. Dynamic range\n"
                     "corresponding to each network tensor is required to run the sample. Defaults to network_tensors.txt."
                    << std::endl;
    std::cout <<    "--data=/path/to/data/dir. Specify data directory to search for above files in case absolute paths to\n"
                    "files are not provided. Defaults to data/samples/int8_api/ or data/int8_api/."
                    << std::endl;
    std::cout <<    "--useDLACore=N. Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1,\n"
                    "where n is the number of DLA engines on the platform."
                    << std::endl;
    std::cout <<    "--topBottomK=K. Specify how many Top-K results shall be output. Both the Top-K and the Bottom-K predictions\n"
                    "will be printed with their output activation values. Defaults to 5 (for Top-5 and Bottom-5 results)."
                    << std::endl;
    std::cout <<    "--fp32. Run inference at FP32 precision on GPU. Cannot be combined with --useDLACore=N (N>=0).\n"
                    "Defaults to running inference at INT8 precision (--fp32 not set)."
                    << std::endl;
    std::cout <<    "--safeGpuInt8. Run inference in safe mode on GPU at INT8. Cannot be combined with --useDLACore=N (N>=0)\n"
                    "and/or --fp32. Defaults to running inference in unsafe mode (--safeGpuInt8 not set)."
                    << std::endl;
    std::cout <<    "--verbose. Outputs per tensor dynamic range and layer precision info for the network." << std::endl;
}

int main(int argc, char** argv)
{
    SampleINT8APIArgs args;
    bool argsOK = parseSampleINT8APIArgs(args, argc, argv);

    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (args.verbose)
    {
        gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleINT8APIParams params;
    params = initializeSampleParams(args);

    SampleINT8API sample(params);
    std::string device = params.dlaCore < 0 ? "GPU" : "DLA";
    std::string precision = params.fp32 ? "FP32" : "INT8";
    gLogInfo << "Building and running a " << precision << " inference engine on " << device
             << " for " << params.modelFileName << std::endl;

    auto buildStatus = sample.build();
    if (buildStatus == Logger::TestResult::kWAIVED)
    {
        return gLogger.reportWaive(sampleTest);
    }
    else if (buildStatus == Logger::TestResult::kFAILED)
    {
        return gLogger.reportFail(sampleTest);
    }

    if (sample.infer() != Logger::TestResult::kRUNNING)
    {
        return gLogger.reportFail(sampleTest);
    }

    if (sample.teardown() != Logger::TestResult::kRUNNING)
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
