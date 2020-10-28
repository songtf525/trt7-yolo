//
// Created by songtf on 20-10-23.
//

#ifndef TRT7_YOLO_YOLOV3LAYER_H
#define TRT7_YOLO_YOLOV3LAYER_H

#include <assert.h>
#include <string>
#include <iostream>
#include "NvInfer.h"
#include "cublas_v2.h"
#include <vector>




struct YoloKernel
{
    int width;
    int height;
    float anchors[3 * 2];
};


template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val)
{
    val = *reinterpret_cast<T*>(buffer);
    buffer += sizeof(T);
}


class Yolov3LayerPlugin : public nvinfer1::IPluginV2IOExt
{
public:
    explicit Yolov3LayerPlugin();

    Yolov3LayerPlugin(const void *data, size_t length);

    ~Yolov3LayerPlugin();

    int getNbOutputs() const override
    {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims) override;

    int initialize() override;

    virtual void terminate() override
    {

    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

    virtual size_t getSerializationSize() const override;

    virtual void serialize(void *buffer) const override;

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const override
    {
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    const char *getPluginType() const override;

    const char *getPluginVersion() const override;

    void destroy() override;

    nvinfer1::IPluginV2IOExt *clone() const override;

    void setPluginNamespace(const char *pluginNamespace) override;

    const char *getPluginNamespace() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, nvinfer1::IGpuAllocator *gpuAllocator) override;

    void configurePlugin(const nvinfer1::PluginTensorDesc *in, int nbInput, const nvinfer1::PluginTensorDesc *out, int nbOutput) override;

    void detachFromContext() override;

private:
    void forwardGPU(const float *const *inputs, float *output, cudaStream_t stream, int batchSize = 1);

    int mClassCount;
    int mKernelCount;
    std::vector<YoloKernel> mYoloKernel;
    int mThreadCount = 256;
    const char *mPluginNamespace;
};

class Yolov3PluginCreator : public nvinfer1::IPluginCreator
{
public:
    Yolov3PluginCreator();

    ~Yolov3PluginCreator() override = default;

    const char *getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection *getFieldNames() override;

    nvinfer1::IPluginV2IOExt *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override;

    nvinfer1::IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

    void setPluginNamespace(const char *libNamespace) override;

    const char *getPluginNamespace() const override;


private:

    std::string mNamespace;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(Yolov3PluginCreator);


#endif //TRT7_YOLO_YOLOV3LAYER_H
