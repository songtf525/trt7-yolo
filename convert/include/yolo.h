//
// Created by songtf on 20-10-23.
//

#ifndef TRT7_YOLO_YOLO_H
#define TRT7_YOLO_YOLO_H


#include "NvInfer.h"
#include "yolov3layer.h"


struct NetworkInfo
{
    std::string networkType;
    std::string configFilePath;
    std::string wtsFilePath;
    std::string labelsFilePath;
    std::string precision;
    std::string deviceType;
    std::string enginePath;
    std::string inputBlobName;
    std::string data_path;
};

struct InferParams
{
    bool printPerfInfo;
    bool printPredictionInfo;
    std::string calibImages;
    std::string calibImagesPath;
    float probThresh;
    float nmsThresh;
};


class Yolo
{
public:
    Yolo(const NetworkInfo& networkInfo, const InferParams& inferParams);

    ~Yolo();

    std::string getNetworkType() const;

    float getNMSThresh() const;

    std::string getClassName(const int& label) const;

    uint32_t getInputH() const;

    uint32_t getInputW() const;

    uint32_t getNumClasses() const;

    void doInference(const unsigned char* input, const uint32_t batchSize);

private:

    void createYOLOEngine(const nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT, Int8EntropyCalibrator* calibrator = nullptr);

    void parseConfigBlocks();

    void allocateBuffers();

    bool verifyYoloEngine();

    void destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights);

    void writePlanFileToDisk();


    const std::string m_NetworkType;
    const std::string m_ConfigFilePath;
    const std::string m_WtsFilePath;
    const std::string m_DeviceType;

};






#endif //TRT7_YOLO_YOLO_H
