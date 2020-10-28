#include "yolov3layer.h"



__device__ float Logist(float data)
{
    return 1.0f / (1.0f + expf(-data));
}

__global__void CalDetection(const float *input,
                            float *output,
                            int noElements,
                            int yoloWidth,
                            int yoloHeight,
                            const float anchors[3*2],
                            int classes, int outputElem)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= noElements)
        return;

    int total_grid = yoloWidth * yoloHeight;
    int bnIdx = idx / total_grid;
    idx = idx - total_grid * bnIdx;
    int info_len_i = 5 + classes;
    const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

    for (int k = 0; k < 3; ++k)
    {
        int class_id = 0;
        float max_cls_prob = 0.0;
        for (int i = 5; i < info_len_i; ++i)
        {
            float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
            if (p > max_cls_prob)
            {
                max_cls_prob = p;
                class_id = i - 5;
            }
        }
        float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
        if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH)
            continue;

        float *res_count = output + bnIdx*outputElem;
        int count = (int)atomicAdd(res_count, 1);

        if (count >= MAX_OUTPUT_BBOX_COUNT)
            return;

        char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
        Detection* det =  (Detection*)(data);

        int row = idx / yoloWidth;
        int col = idx % yoloWidth;

        //Location
        det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
        det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
        det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
        det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
        det->det_confidence = box_prob;
        det->class_id = class_id;
        det->class_confidence = max_cls_prob;
    }

}