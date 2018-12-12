#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
using namespace cv;
using namespace cv::cuda;


void alpha_cuda_caller(const PtrStepSz<float>& shiftMat,
    PtrStep<uchar4> dst,
    cudaStream_t stream);

void alpha_cuda(const cuda::GpuMat& shiftMat,
    cuda::GpuMat& dst,
    cuda::Stream& stream = cuda::Stream::Null()){

    cudaStream_t s = StreamAccessor::getStream(stream);
    alpha_cuda_caller(shiftMat,dst,s);
}    



void combine_caller(const PtrStepSz<uchar4>& imageL,
    const PtrStepSz<uchar4>& imageR,
    const PtrStepSz<float>& flowMagL,
    const PtrStepSz<float>& flowMagR,
    PtrStep<uchar4> dst,cudaStream_t stream);

void combine_cuda(const cuda::GpuMat& imageL,
    const cuda::GpuMat& imageR,
    const cuda::GpuMat& flowMagL,
    const cuda::GpuMat& flowMagR,
    cuda::GpuMat& dst,
    cuda::Stream& stream = cuda::Stream::Null()){
    dst.create(imageL.size(),imageL.type());
    cudaStream_t s = StreamAccessor::getStream(stream);
    combine_caller(imageL,imageR,flowMagL,flowMagR, dst,s);
}

void feather_alpha_chanel_caller(
	const PtrStepSz<uchar4>& src,
	PtrStep<uchar4> dst,
	const int feathersize,
	cudaStream_t stream);

void feather_alpha_chanel(
	const cuda::GpuMat& src,
	cuda::GpuMat& dst,
	const int feathersize,
	cuda::Stream& stream = cuda::Stream::Null())
{
	dst.create(src.size(), src.type());
	cudaStream_t s = StreamAccessor::getStream(stream);
	feather_alpha_chanel_caller(src, dst, feathersize, s);
}

void extendimage_caller(const PtrStepSz<uchar4>& gpucroppedSideSpherical,
	PtrStep<uchar4> gpuextendedSideSpherical,
	const PtrStepSz<uchar4>& gpufisheyeSpherical,
	PtrStep<uchar4> gpuextendedFisheyeSpherical,
	const int extendedWidth,
	cudaStream_t stream);

void extendimage(
	const cuda::GpuMat& gpucroppedSideSpherical,
	cuda::GpuMat& gpuextendedSideSpherical,
	const cuda::GpuMat& gpufisheyeSpherical,
	cuda::GpuMat& gpuextendedFisheyeSpherical,
	const int extendedWidth,
	cuda::Stream& stream = cuda::Stream::Null())
{
	gpuextendedSideSpherical.create(Size(extendedWidth, gpufisheyeSpherical.rows), CV_8UC4);
	gpuextendedFisheyeSpherical.create(gpuextendedSideSpherical.size(), CV_8UC4);
	cudaStream_t s = StreamAccessor::getStream(stream);
	extendimage_caller(gpucroppedSideSpherical,
		gpuextendedSideSpherical,
		gpufisheyeSpherical,
		gpuextendedFisheyeSpherical,
		extendedWidth,
		s);
}

void flatten_caller(
	const PtrStepSz<uchar4>& bottomLayer,
	const PtrStepSz<uchar4>& topLayer,
	PtrStep<uchar4> dst,
	cudaStream_t stream);

void Gpu_flatten(
	const cuda::GpuMat& bottomLayer,
	const cuda::GpuMat& topLayer,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null())
{
	dst.create(bottomLayer.size(), bottomLayer.type());
	cudaStream_t s = StreamAccessor::getStream(stream);
	flatten_caller(bottomLayer, topLayer, dst, s);
}

void strip_caller(
	const PtrStepSz<uchar4> fisheyeSpherical,
	const PtrStepSz<uchar4> warpedExtendedFisheyeSpherical,
	const PtrStepSz<uchar4> warpedSphericalForEye,
	PtrStep<uchar4> imgdst,
	cudaStream_t stream);

void Gpu_strip(
	const cuda::GpuMat& fisheyeSpherical,
	const cuda::GpuMat& warpedExtendedFisheyeSpherical,
	const cuda::GpuMat& warpedSphericalForEye,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null())
{
	dst.create(warpedSphericalForEye.size(), warpedSphericalForEye.type());
	cudaStream_t s = StreamAccessor::getStream(stream);
	strip_caller(fisheyeSpherical, warpedExtendedFisheyeSpherical, warpedSphericalForEye, dst, s);
}


