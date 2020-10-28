//
// Created by songtf on 20-10-23.
//


#include "yolo.h"
#include <sstream>
#include <fstream>

Yolo::Yolo(const NetworkInfo &networkInfo, const InferParams &inferParams) :
        m_NetworkType(networkInfo.networkType),
        m_ConfigFilePath(networkInfo.configFilePath),
        m_WtsFilePath(networkInfo.wtsFilePath),
        m_DeviceType(networkInfo.deviceType)
{

    m_configBlocks = parseConfigFile(m_ConfigFilePath);  // 解析配置网络

    if (m_NetworkType == "yolov5")
    {
        parse_cfg_blocks_v5(m_configBlocks);
    }

    // engine Path
    m_EnginePath = networkInfo.data_path + "-" + m_Precision + "-batch" + std::to_string(m_BatchSize) + ".engine";

    if (m_Precision == "kFLOAT")
    {
        createYOLOEngine();
    } else if (m_Precision == "kINT8")
    {
        Int8EntropyCalibrator calibrator(m_BatchSize,
                                         m_CalibImages,
                                         m_CalibImagesFilePath,
                                         m_CalibTableFilePath, m_InputSize, m_InputH, m_InputW,
                                         m_InputBlobName);

        createYOLOEngine(nvinfer1::DataType::kINT8, &calibrator);
    } else if (m_Precision == "kHALF")
    {
        createYOLOEngine(nvinfer1::DataType::kHALF, nullptr);
    } else
    {
        std::cout << "Unrecognized precision type " << m_Precision << std::endl;
    }


}


Yolo::~Yolo()
{
//    for (auto& tensor : m_OutputTensors)
//    {
//        NV_CUDA_CHECK(cudaFreeHost(tensor.hostBuffer));
//    }
//
//    for(auto& deviceBuffer ： m_DeviceBuffers)
//    {
//        NV_CUDA_CHECK(cudaFree(deviceBuffer));
//    }
}


void Yolo::createYOLOEngine(const nvinfer1::DataType dataType, Int8EntropyCalibrator *calibrator)
{
    if (fileExists(m_EnginePath))
        return;

    std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);

    std::vector<nvinfer1::Weights> trtWeights;

    int weightPtr = 0;
    int channels = m_InputChannel;


    m_Builder = nvinfer1::createInferBuilder(m_Logger);

    nvinfer1::IBuilderConfig *config = m_Builder->createBuilerConfig();
    m_Network = m_Builder->createNetworkV2(0U);

    if ((dataType == nvinfer1::DataType::kINT8 && !m_Builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !m_Builder->platformHasFastFp16()))
    {
        std::cout << "Platform doesn't support this precision." << std::endl;
        assert(0);
    }


    // Input
    nvinfer1::ITensor *data = m_Network->addInput(m_InputBlobName.c_str(),
                                                  nvinfer1::DataType::kFLOAT,
                                                  nvinfer1::Dims{static_cast<int>(m_InputChannel), static_cast<int>(m_InputHeight),
                                                                 static_cast<int>(m_InputWidth)});


    assert(data != nullptr);

    // Add elementwise layer to normalize pixel value 0-1
    nvinfer1::Dims divDims
            {
                    3,
                    {static_cast<int>(m_InputChannel), static_cast<int>(m_InputHeight), static_cast<int>(m_InputWidth)},
                    {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
                     nvinfer1::DimensionType::kSPATIAL}
            };


    nvinfer1::Weights divWeights
            {
                    nvinfer1::DataType::kFLOAT, nullptr,
                    static_cast<int64_t>(m_InputSize)
            };

    float *divWt = new float[m_InputSize];
    for (uint32_t w = 0; w < m_InputSize; ++w)
    {
        divWt[w] = 255.0;
    }

    divWeights.values = divWt;
    trtWeights.push_back(divWeights);
    nvinfer1::IConstantLayer *constDivide = m_Network->addConstant(divDims, divWeights);
    assert(constDivide != nullptr);


    nvinfer1::IElementWiseLayer *elementDivide = m_Network->addElementWise(
            *data, *constDivide->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
    assert(elementDivide != nullptr);

    nvinfer1::ITensor *previous = elementDivide->getOutput(0);
    std::vector<nvinfer1::ITensor *> tensorOutputs;
    uint32_t outputTensorCount = 0;

    ///



    for (uint32_t i = 0; i < m_configBlocks.size(); ++i)
    {
        // check if num of channels is correct
        assert(getNumChannels(previout) == channels);

        std::string layerIndex = "(" + std::to_string(i) + ")";
        if (m_configBlocks.at(i).at("type") == "net")
        {
            // printLayerInfo
            printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
        } else if (m_configBlocks.at(i).at("type") == "convolutional")
        {

            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer *out;
            std::string layerType;

            std::string activation = "";
            if (m_configBlocks.at(i).find("activation") != m_configBlocks.at(i).end())
            {
                activation = m_configBlocks[i]["activation"];
            }
            // check if batch_norm enabled
            if ((m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end() &&
                 ("leaky") == activation))
            {
                out = netAddConvBNLeaky(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                        channels, previous, m_Network);
                layerType = "conv-bn-leaky";
            } else if ((m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end() &&
                        ("mish") == activation))
            {
                out = net_conv_bn_mish(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                       channels, previous, m_Network);
                layerType = "conv-bn-mish";
            } else
            {
                out = netAddConvLinear(i, m_configBlocks.at(i), weights, trtWeights, weightPtr,
                                       channels, previous, m_Network);
                layerType = "conv-linear";
            }


            previous = out->getOutput(0);
            assert(previous != nullptr);

            channels = getNumChannels(previous);

            std::string outputVol = dimsTostring(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));

            // print
            printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));

        } else if (m_configBlocks.at(i).at("type") == "shortcut")
        {
            assert(m_configBlocks.at(i).at("activation") == "linear");
            assert(m_configBlocks.at(i).find("from") != m_configBlocks.at(i).end());

            int from = std::stoi(m_configBlocks.at(i).at("from"));

            std::string inputVol = dimsToString(previous->getDimensions());

            // check if indexes are correct
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);

            nvinfer1::IElementWiseLayer *ew = m_Network->addElementWise(
                    *tensorOutputs[i - 2],
                    *tensorOutputs[i + from - 1],
                    nvinfer1::ElementWiseOperation::kSUM);

            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());

            previous = ew->getOutput(0);
            assert(previous != nullptr);

            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));

            // print
            printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
        } else if (m_configBlocks.at(i).at("type") == "route")
        {
            size_t found = m_configBlocks.at(i).at("layers").find(",");
            if (found != std::string::npos)  // concate multi layers
            {
                std::vector<int> vec_index = split_layer_index(m_configBlocks.at(i).at("layers"), ",");
                for (auto &ind_layer:vec_index)
                {
                    if (ind_layer < 0)
                    {
                        ind_layer = static_cast<int>(tensorOutputs.size()) + ind_layer;
                    }
                    assert(ind_layer < static_cast<int>(tensorOutputs.size()) && ind_layer >= 0);
                }

                nvinfer1::ITensor **concatInputs = reinterpret_cast<nvinfer1::ITensor **>(malloc(sizeof(nvinfer1::ITensor *)
                                                                                                 * vec_index.size()));
                for (size_t ind = 0; ind < vec_index.size(); ++ind)
                {
                    concatInputs[ind] = tensorOutputs[vec_index[ind]];
                }

                nvinfer1::IConcatenationLayer *concat = m_Network->addConcatenation(concatInputs, static_cast<int>(vec_index.size()));
                assert(concat != nullptr);

                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);

                nvinfer1::Dims debug = previous->getDimensions();
                std::string outputVol = dimsToString(previous->getDimensions());
                int nums = 0;
                for (auto &indx: vec_index)
                {
                    nums += getNumChannels(tensorOutputs[indx]);
                }
                channels = nums;

                tensorOutputs.push_back(concat->getOutput(0));
                // print
                printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));
            } else  // single layer
            {
                int idx = std::stoi(trim(m_configBlocks.at(i).at("layers")));
                if (idx < 0)
                {
                    idx = static_cast<int>(tensorOutputs.size()) + idx;
                }
                assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);

                // route
                if (m_configBlocks.at(i).find("groups") == m_configBlocks.at(i).end())
                {
                    previous = tensorOutputs[idx];
                    assert(previous != nullptr);
                    std::string outputVol = dimsToString(previous->getDimensions());
                    // set the output volume depth
                    channels = getNumChannels(tensorOutputs[idx]);
                    tensorOutputs.push_back(tensorOutputs[idx]);
                    printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));
                } else // yolov4-tiny route split layer
                {
                    if (m_configBlocks.at(i).find("group_id") == m_configBlocks.at(i).end())
                    {
                        assert(0);
                    }
                    int chunk_idx = std::stoi(trim(m_configBlocks.at(i).at("group_id")));
                    nvinfer1::ILayer *out = layer_split(i, tensorOutputs[idx], m_Network);
                    std::string inputVol = dimsToString(previous->getDimensions());
                    previous = out->getOutput(chunk_idx);
                    assert(previous != nullptr);
                    channels = getNumChannels(previous);
                    std::string outputVol = dimsToString(previous->getDimensions());
                    tensorOutputs.push_back(out->getOutput(chunk_idx));
                    printLayerInfo(layerIndex, "chunk", inputVol, outputVol, std::to_string(weightPtr));
                }
            }
        } else if (m_configBlocks.at(i).at("type") == "upsample")
        {
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer *out = netAddUpsample(i - 1, m_configBlocks[i], weights, trtWeights,
                                                   channels, previous, m_Network);
            previous = out->getOutput(0);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
        } else if (m_configBlocks.at(i).at("type") == "maxpool")
        {
            // Add same padding layers
            if (m_configBlocks.at(i).at("size") == "2" && m_configBlocks.at(i).at("stride") == "1")
            {
                m_TinyMaxpoolPaddingFormula->addSamePaddingLayer("maxpool_" + std::to_string(i));
            }
            std::string inputVol = dimsToString(previous->getDimensions());
            nvinfer1::ILayer *out = netAddMaxpool(i, m_configBlocks.at(i), previous, m_Network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = dimsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
            printLayerInfo(layerIndex, "maxpool", inputVol, outputVol, std::to_string(weightPtr));
        } else if (m_configBlocks.at(i).at("type") == "yolo")
        {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
            // assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo &curYoloTensor = m_OutputTensors.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.grid_h = prevTensorDims.d[1];
            curYoloTensor.grid_w = prevTensorDims.d[2];
            curYoloTensor.stride = m_InputW / curYoloTensor.gridSize;
            curYoloTensor.stride_h = m_InputH / curYoloTensor.grid_h;
            curYoloTensor.stride_w = m_InputW / curYoloTensor.grid_w;
            m_OutputTensors.at(outputTensorCount).volume = curYoloTensor.grid_h
                                                           * curYoloTensor.grid_w
                                                           * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(outputTensorCount);
            curYoloTensor.blobName = layerName;
            nvinfer1::IPlugin *yoloPlugin
                    = new YoloLayerV3(m_OutputTensors.at(outputTensorCount).numBBoxes,
                                      m_OutputTensors.at(outputTensorCount).numClasses,
                                      m_OutputTensors.at(outputTensorCount).grid_h,
                                      m_OutputTensors.at(outputTensorCount).grid_w);
            assert(yoloPlugin != nullptr);
            auto yolo = m_Network->addPluginV2(&previous, 1, *yoloPlugin);
            assert(yolo != nullptr);
            yolo->setName(layerName.c_str());

            std::string inputVol = dimsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = dimsToString(previous->getDimensions());
            m_Network->markOutput(*yolo->getOutput(0));
            channels = getNumChannels(yolo->getOutput(0));
            tensorOutputs.push_back(yolo->getOutput(0));

            printLayerInfo(layerIndex, "yolo", inputVol, outputVol, std::to_string(weightPtr));
            ++outputTensorCount;
        } else
        {
            std::cout << "Unsupported layer type --> \"" << m_configBlocks.at(i).at("type") << "\""
                      << std::endl;
            assert(0);
        }
    }


    if (static_cast<int>(weights.size()) != weightPtr)
    {
        std::cout << "Number of unused weights left : " << static_cast<int>(weights.size()) - weightPtr << std::endl;
        assert(0);
    }

//        std::cout << "Output blob names: " << std::endl;
//        for(auto & tensor: m_OuputTensors)
//        {
//            std::cout<< tensor.blobName << std::endl;
//        }

    // Create and cache the engine if not already present
    if (fileExists(m_EnginePath))
    {
        std::cout << "Using previously generated plan file located at " << m_EnginePath << std::endl;
        destroyNetworkUtils(trtWeights);
        return;
    }

    m_Builder->setMaxBatchSize(m_BatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    if (dataType == nvinfer1::DataType::kINT8)
    {
        assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
    } else if (dataType == nvinfer1::DataType::kHALF)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    m_Builder->allowGPUFallback(true);

    int nbLayers = m_Network->getNbLayers();
    int layersOnDLA = 0;

    for (int i = 0; i < nbLayers; i++)
    {
        nvinfer1::ILayer *curLayer = m_Network->getLayer(i);
        if (m_DeviceType == "kDLA" && m_Builder->canRunOnDLA(curLayer))
        {
            m_Builder->setDeviceType(curLayer, nvinfer1::DeviceType::kDLA);
            layersOnDLA++;
            std::cout << "Set layer " << curLayer->getName() << " to run on DLA" << std::endl;
        }
    }

    // Build the engine
    std::cout << "Building the TensorRT Engine..." << std::endl;
    m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *config);
    assert(m_Engine != nullptr);
    std::cout << "Building complete" << std::endl;

    /// Serialize the engine
    writePlanFileToDisk();

    // destroy
    destroyNetworkUtils(trtWeights);
}


void Yolo::allocateBuffers()
{
    m_deviceBuffers.resize(m_Engine->getNbBindings(), nullptr);
    assert(m_InputBindingIndex != -1 && "Invalid input binding index");
    cudaMalloc(&m_DeviceBuffers.at(m_InputBindingIndex), m_BatchSize * m_InputSize * sizeof(float));

    for(auto &tensor : m_OutputTensors)
    {
        tensor.bindingIndex = m_Engine->getBindingIndex(tensor.blobName.c_str());
        assert((tensor.bindingIndex != -1) && "Invalid output binding index");

        NV_CUDA_CHECK(cudaMalloc(&m_DeviceBuffers.at(tensor.bindingIndex),
                                 m_BatchSize * tensor.volume * sizeof(float)));
        NV_CUDA_CHECK(
                cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    }
}


bool Yolo::verifyYoloEngine()
{
    assert((m_Engine->getNbBindings() == (1 + m_OutputTensors.size())
            && "Binding info doesnot match between cfg and engine file \n"));

    for(auto tensor : m_OutputTensors)
    {
        assert(!strcmp(m_Engine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str())
               && "Blobs names dont match between cfg and engine file \n");
        assert(get3DTensorVolume(m_Engine->getBindingDimensions(tensor.bindingIndex))
               == tensor.volume
               && "Tensor volumes dont match between cfg and engine file \n");
    }
    assert(m_Engine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
    assert(m_Engine->getBindingName(m_InputBindingIndex) == m_InputBlobName
           && "Input blob name doesn't match between config and engine file");
    assert(get3DTensorVolume(m_Engine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);
    return true;
}

void Yolo::destroyNetworkUtils(std::vector<nvinfer1::Weights> &trtWeights)
{
    if (m_Network)
    {
        m_Network->destroy();
    }
    if (m_Engine)
    {
        m_Engine->destroy();
    }
    if (m_Builder)
    {
        m_Builder->destroy();
    }
    if (m_ModelStream)
    {
        m_ModelStream->destroy();
    }

    // deallocate the weights
    for (uint32_t i = 0; i < trtWeights.size(); ++i)
    {
        if (trtWeights[i].count > 0)
        {
            free(const_cast<void *>(trtWeights[i].values));
        }
    }
}


void Yolo::writePlanFileToDisk()
{
    std::cout << "serializing the TensorRT Engine ..." << std::endl;
    assert(m_Engine && "Invalid TensorRT Engine");
    m_ModelStream = m_Engine->serialize();
    assert(m_ModelStream && "Unable to serialize engine");
    assert(!m_EnginePath.empty() && "Enginepath is empty");

    // write data to output file
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write(static_cast<const char *>(m_ModelStream->data()), m_ModelStream->size());
    std::ofstream outFile;

    outFile.open(m_EnginePath, std::ios::binary | std::ios::out);
    outFile << gieModelStream.rdbuf();
    outFile.close();

    std::cout << "serialized plan file cached at location: " << m_EnginePath << std::endl;
}