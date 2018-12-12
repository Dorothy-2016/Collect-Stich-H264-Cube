#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudev/common.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector_types.h>
using namespace cv;
using namespace cv::cuda;
//自定义内核函数

__global__ void combine_kernel(const PtrStepSz<uchar4> imageL,
    const PtrStepSz<uchar4> imageR,
    const PtrStepSz<float> flowMagL,
    const PtrStepSz<float> flowMagR,
    PtrStep<uchar4> dst){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < imageL.cols && y < imageL.rows)
    {
        uchar4 colorL = imageL(y,x);
        uchar4 colorR = imageR(y,x);
        unsigned char outAlpha;
        if(colorL.w > colorR.w)
        {
            if(colorL.w / 255.0f > 0.1)
                outAlpha = 255;
            else
                outAlpha = 0;
        }
        else
        {
            if(colorR.w / 255.0f > 0.1)
                outAlpha = 255;
            else
                outAlpha = 0;
        }
        uchar4 colorMixed;
        if (colorL.w == 0 && colorR.w == 0) {
            colorMixed = make_uchar4(0, 0, 0, outAlpha);
        } else if (colorL.w == 0) {
            colorMixed = make_uchar4(colorR.x, colorR.y, colorR.z, outAlpha);
        } else if (colorR.w == 0) {
            colorMixed = make_uchar4(colorL.x, colorL.y, colorL.z, outAlpha);
        } else {
            const float magL = flowMagL(y,x) / float(imageL.cols);
            const float magR = flowMagR(y,x) / float(imageL.cols);
            float blendL = float(colorL.w);
            float blendR = float(colorR.w);
            float norm = blendL + blendR;
            blendL /= norm;
            blendR /= norm;
            const float colorDiff =
              (abs(colorL.x - colorR.x) +
               abs(colorL.y - colorR.y) +
               abs(colorL.z - colorR.z)) / 255.0f;
            const float kColorDiffCoef = 10.0f;
            const float kSoftmaxSharpness = 10.0f;
            const float kFlowMagCoef = 20.0f; // NOTE: this is scaled differently than the test version due to normalizing magL & magR by imageL.cols
            const float deghostCoef = tanhf(colorDiff * kColorDiffCoef);
            const double expL = exp(kSoftmaxSharpness * blendL * (1.0 + kFlowMagCoef * magL));
            const double expR = exp(kSoftmaxSharpness * blendR * (1.0 + kFlowMagCoef * magR));
            const double sumExp = expL + expR + 0.00001;
            const float softmaxL = float(expL / sumExp);
            const float softmaxR = float(expR / sumExp);
            colorMixed = make_uchar4(
                float(colorL.x)* (blendL * (1-deghostCoef) + softmaxL * deghostCoef) + float(colorR.x)*(blendR * (1-deghostCoef) + softmaxR * deghostCoef),
                float(colorL.y)* (blendL * (1-deghostCoef) + softmaxL * deghostCoef) + float(colorR.y)*(blendR * (1-deghostCoef) + softmaxR * deghostCoef),
                float(colorL.z)* (blendL * (1-deghostCoef) + softmaxL * deghostCoef) + float(colorR.z)*(blendR * (1-deghostCoef) + softmaxR * deghostCoef),  
                255);         
        }
        dst(y, x) = colorMixed;
        // uchar4 v = imageL(y,x);
        // dst(y,x) = make_uchar4(v.x,v.y,v.z,255);
    }
}

void combine_caller(const PtrStepSz<uchar4>& imageL,
    const PtrStepSz<uchar4>& imageR,
    const PtrStepSz<float>& flowMagL,
    const PtrStepSz<float>& flowMagR,
    PtrStep<uchar4> dst,cudaStream_t stream){
    dim3 block(32,8);
    dim3 grid((imageL.cols + block.x - 1)/block.x,(imageL.rows + block.y - 1)/block.y);

    combine_kernel<<<grid,block,0,stream>>>(imageL,imageR,flowMagL,flowMagR, dst);
    if(stream == 0)
        cudaDeviceSynchronize();
}

__global__ void shift_kernel(const PtrStepSz<float> shiftMat,
    PtrStep<uchar4> dst){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < shiftMat.cols && y < shiftMat.rows)
    {
        uchar4 image = dst(y,x);
        float alpha = shiftMat(y,x);
        image.w = (int)(image.w * alpha);
        dst(y,x) = image;
    }
}


void alpha_cuda_caller(const PtrStepSz<float>& shiftMat,
    PtrStep<uchar4> dst,
    cudaStream_t stream){
    dim3 block(32,8);
    dim3 grid((shiftMat.cols + block.x - 1)/block.x,(shiftMat.rows + block.y - 1)/block.y);

    shift_kernel<<<grid,block,0,stream>>>(shiftMat,dst);
    if(stream == 0)
        cudaDeviceSynchronize();
}

__global__ void feather_alpha_chanel_kernel(
	const PtrStepSz<uchar4> src,
	PtrStep<uchar4> dst,
	const int feathersize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int yFeatherStart = src.rows - 1 - feathersize;
	if (x < src.cols && y < src.rows)
	{
		uchar4 gsrc = src(y, x);
		const float alpha =
			1.0f - float(y - yFeatherStart) / float(feathersize);

		uchar4 colorMixed;
		unsigned char temp = gsrc.w <= (unsigned char)(255.0f * alpha) ? gsrc.w : (unsigned char)(255.0f * alpha);
		if (y >= yFeatherStart)
		{
			colorMixed = make_uchar4(
				gsrc.x,
				gsrc.y,
				gsrc.z,
				temp
				);
		}
		else
		{
			colorMixed = make_uchar4(
				gsrc.x,
				gsrc.y,
				gsrc.z,
				gsrc.w
				);
		}
		dst(y, x) = colorMixed;
	}
}

void feather_alpha_chanel_caller(
	const PtrStepSz<uchar4>& src,
	PtrStep<uchar4> dst,
	const int feathersize,
	cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

	feather_alpha_chanel_kernel <<<grid, block, 0, stream >>>(src, dst, feathersize);
	if (stream == 0)
		cudaDeviceSynchronize();
}

__global__ void extendimage_kernal(const PtrStepSz<uchar4> gpucroppedSideSpherical,
	PtrStep<uchar4> gpuextendedSideSpherical,
	const PtrStepSz<uchar4> gpufisheyeSpherical,
	PtrStep<uchar4> gpuextendedFisheyeSpherical,
	const int extendedWidth)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	const int width = gpufisheyeSpherical.cols;
	if (x < extendedWidth && y < gpufisheyeSpherical.rows)
	{
		uchar4 g1 = gpucroppedSideSpherical(y, x % width);
		uchar4 g2 = gpufisheyeSpherical(y, x % width);

		gpuextendedSideSpherical(y, x) = g1;
		gpuextendedFisheyeSpherical(y, x) = g2;
	}
}

void extendimage_caller(const PtrStepSz<uchar4>& gpucroppedSideSpherical,
	PtrStep<uchar4> gpuextendedSideSpherical,
	const PtrStepSz<uchar4>& gpufisheyeSpherical,
	PtrStep<uchar4> gpuextendedFisheyeSpherical,
	const int extendedWidth,
	cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((extendedWidth + block.x - 1) / block.x, (gpucroppedSideSpherical.rows + block.y - 1) / block.y);
	extendimage_kernal << <grid, block, 0, stream >> >(gpucroppedSideSpherical, gpuextendedSideSpherical, gpufisheyeSpherical, gpuextendedFisheyeSpherical, extendedWidth);
	if (stream == 0)
		cudaDeviceSynchronize();
}


__global__ void flatten_kernel(
	const PtrStepSz<uchar4> bottomLayer,
	const PtrStepSz<uchar4> topLayer,
	PtrStep<uchar4> dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < bottomLayer.cols && y < bottomLayer.rows && x >= 0 && y >= 0)
	{
		uchar4 baseColor = bottomLayer(y, x);
		uchar4 topColor;
		if (x < topLayer.cols && y < topLayer.rows)
			topColor = topLayer(y, x);
		else
			topColor = make_uchar4(0.0, 0.0, 0.0, 0.0);

		const float colorDiff =
			(abs(baseColor.x - topColor.x) +
				abs(baseColor.y - topColor.y) +
				abs(baseColor.z - topColor.z)) / 255.0f;

		static const float kColorDiffCoef = 5.0f;
		static const float kSoftmaxSharpness = 5.0f;
		static const float kBaseLayerBias = 2.0f;
		const float deghostCoef = tanhf(colorDiff * kColorDiffCoef);
		const float alphaR = topColor.w / 255.0f;
		const float alphaL = 1.0f - alphaR;
		const double expL = exp(kSoftmaxSharpness * alphaL * kBaseLayerBias);
		const double expR = exp(kSoftmaxSharpness * alphaR);
		const double sumExp = expL + expR + 0.00001;
		const float softmaxL = float(expL / sumExp);
		const float softmaxR = 1.0f - softmaxL;

		unsigned char outAlpha;
		if (topColor.w >= baseColor.w)
			outAlpha = topColor.w;
		else
			outAlpha = baseColor.w;

		uchar4 colorMixed;
		colorMixed = make_uchar4(
			float(baseColor.x) * (alphaL * (1 - deghostCoef) + softmaxL * deghostCoef) + float(topColor.x) * (alphaR * (1 - deghostCoef) + softmaxR * deghostCoef),
			float(baseColor.y) * (alphaL * (1 - deghostCoef) + softmaxL * deghostCoef) + float(topColor.y) * (alphaR * (1 - deghostCoef) + softmaxR * deghostCoef),
			float(baseColor.z) * (alphaL * (1 - deghostCoef) + softmaxL * deghostCoef) + float(topColor.z) * (alphaR * (1 - deghostCoef) + softmaxR * deghostCoef),
			outAlpha);

		dst(y, x) = colorMixed;
	}
}

void flatten_caller(
	const PtrStepSz<uchar4>& bottomLayer,
	const PtrStepSz<uchar4>& topLayer,
	PtrStep<uchar4> dst,
	cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((bottomLayer.cols + block.x - 1) / block.x, (bottomLayer.rows + block.y - 1) / block.y);

	flatten_kernel << <grid, block, 0, stream >> >(bottomLayer, topLayer, dst);
	if (stream == 0)
		cudaDeviceSynchronize();
}


__global__ void strip_kernel(
	const PtrStepSz<uchar4> fisheyeSpherical,
	const PtrStepSz<uchar4> warpedExtendedFisheyeSpherical,
	const PtrStepSz<uchar4> warpedSphericalForEye,
	PtrStep<uchar4> imgdst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int maxBlendX = float(fisheyeSpherical.cols) * 0.2;

	if (y < warpedSphericalForEye.rows && x < maxBlendX) {
		uchar4 src = warpedSphericalForEye(y, x);
		uchar4 wrap = warpedExtendedFisheyeSpherical(y, x+fisheyeSpherical.cols);
		float alpha = 1.0f - max(0.0f, min(1.0f, (x - float(maxBlendX) * 0.333f) / (float(maxBlendX) * 0.667f - float(maxBlendX) * 0.333f)));

		uchar4 colorMixed = make_uchar4(
			src.x*alpha + wrap.x*(1-alpha),
			src.y*alpha + wrap.y*(1-alpha),
			src.z*alpha + wrap.z*(1-alpha),
			src.w);
		imgdst(y, x) = colorMixed;
	}
}

void strip_caller(
	const PtrStepSz<uchar4> fisheyeSpherical,
	const PtrStepSz<uchar4> warpedExtendedFisheyeSpherical,
	const PtrStepSz<uchar4> warpedSphericalForEye,
	PtrStep<uchar4> imgdst,
	cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((warpedSphericalForEye.cols + block.x - 1) / block.x, (warpedSphericalForEye.rows + block.y - 1) / block.y);

	strip_kernel << <grid, block, 0, stream >> >(fisheyeSpherical, warpedExtendedFisheyeSpherical, warpedSphericalForEye, imgdst);
	if (stream == 0)
		cudaDeviceSynchronize();
}