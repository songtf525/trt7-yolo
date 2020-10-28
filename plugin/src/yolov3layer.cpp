//
// Created by songtf on 20-10-23.
//

#include "yolov3layer.h"

#include <cudnn.h>


Yolov3LayerPlugin::Yolov3LayerPlugin()
{
    mClassCount = CLASS_NUM;
    mYoloKernel.clear();

    mYoloKernel.push_back(yolo1);
    mYoloKernel.push_back(yolo2);
    mYoloKernel.push_back(yolo3);

    mKernelCount = mYoloKernel.size();
}

Yolov3LayerPlugin::~Yolov3LayerPlugin()
{
}

// create the plugin at runtime from a byte stream
Yolov3LayerPlugin::Yolov3LayerPlugin(const void *data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    read(d, mClassCount);
    read(d, mThreadCount);
    read(d, mKernelCount);
    mYoloKernel.resize(mKernelCount);
    auto kernelSize = mKernelCount * sizeof(YoloKernel);
    memcpy(mYoloKernel.data(), d, kernelSize);
    d += kernelSize;

    assert(d == a + length);
}


void Yolov3LayerPlugin::serialize(void *buffer) const
{
    char *d = static_cast<char *>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mThreadCount);
    write(d, mKernelCount);
    auto kernelSize = mKernelCount * sizeof(YoloKernel);
    memcpy(d, mYoloKernel.data(), kernelSize);
    d += kernelSize;
    assert(d == a + getSerializationSize());
}

size_t Yolov3LayerPlugin::getSerializationSize() const
{
    return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(YoloKernel) * mYoloKernel.size();
}

int Yolov3LayerPlugin::initialize()
{
    return 0;
}

nvinfer1::Dims Yolov3LayerPlugin::getOutputDimensions(int index, const nvinfer1::Dims *inputs, int nbInputDims)
{
    // output the result to channel
    int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

    return nvinfer1::Dims3(totalsize + 1, 1, 1);
}

void Yolov3LayerPlugin::setPluginNamespace(const char *pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char *Yolov3LayerPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
nvinfer1::DataType Yolov3LayerPlugin::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const
{
    return nvinfer1::DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool Yolov3LayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that  is broadcast across batach without replication.
bool Yolov3LayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void Yolov3LayerPlugin::configurePlugin(const nvinfer1::PluginTensorDesc *in, int nbInput, const nvinfer1::PluginTensorDesc *out, int nbOutput)
{

}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource
void Yolov3LayerPlugin::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, nvinfer1::IGpuAllocator *gpuAllocator)
{

}

// Detach the plugin object from its execution context.
void Yolov3LayerPlugin::detachFromContext()
{

}

const char *Yolov3LayerPlugin::getPluginType() const
{
    return "YoloLayer_TRT";
}

const char *Yolov3LayerPlugin::getPluginVersion() const
{
    return "1";
}

void Yolov3LayerPlugin::destroy()
{
    delete this;
}

// clone the plugin
nvinfer1::IPluginV2IOExt *Yolov3LayerPlugin::clone() const
{
    Yolov3LayerPlugin *p = new Yolov3LayerPlugin();
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

void Yolov3LayerPlugin::forwardGPU(const float *const *inputs, float *output, cudaStream_t stream, int batchSize)
{
    void *devAnchor;
    size_t AnchorLen = sizeof(float) * 3 * 2;
    cudaMalloc(&devAnchor, AnchorLen);

    int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

    for (int idx = 0; idx < batchSize; ++idx)
    {
        cudaMemset(output + idx * outputElem, 0, sizeof(float));
    }

    int numElem = 0;
    for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
    {
        const auto &yolo = mYoloKernel[i];
        numElem = yolo.width * yolo.height * batchSize;
        if (numElem < mThreadCount)
        {
            mThreadCount = numElem;
        }

        cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice);
        CalDetection << < (yolo.width * yolo.height * batchSize + mThreadCount - 1) / mThreadCount, mThreadCount >> > (inputs[i], output, numElem,
                yolo.width, yolo.height, (float *) devAnchor, mClassCount, outputElem);
    }

    cudaFree(devAnchor);
}


int Yolov3LayerPlugin::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream)
{
    forwardGPU((const float *const *) inputs, (float *) outputs[0], stream, batchSize);
    return 0;
}

nvinfer1::PluginFieldCollection Yolov3PluginCreator::mFC{};
std::vector<nvinfer1::PluginField> Yolov3PluginCreator::mPluginAttributes;

Yolov3PluginCreator::Yolov3PluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *Yolov3PluginCreator::getPluginName() const
{
    return "YoloLayer_TRT";
}

const char *Yolov3PluginCreator::getPluginVersion() const
{
    return "1";
}


const nvinfer1::PluginFieldCollection *Yolov3PluginCreator::getFieldNames()
{
    return &mFC;
}

nvinfer1::IPluginV2IOExt *Yolov3PluginCreator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
{
    Yolov3LayerPlugin *obj = new Yolov3LayerPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

nvinfer1::IPluginV2IOExt *Yolov3PluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will call Mishplugin::destroy()
    Yolov3LayerPlugin *obj = new Yolov3LayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void Yolov3PluginCreator::setPluginNamespace(const char *libNamespace)
{
    mNamespace = libNamespace;
}

const char *Yolov3PluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}