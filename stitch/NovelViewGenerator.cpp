#include "NovelViewGenerator.h"
using namespace cv;

void combine_cuda(const cuda::GpuMat& imageL,
	const cuda::GpuMat& imageR,
	const cuda::GpuMat& flowMagL,
	const cuda::GpuMat& flowMagR,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null());

void alpha_cuda(const cuda::GpuMat& shiftMat,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null());

void NovelViewGenerator::prepare(
	const cuda::GpuMat& colorImageL,
	const cuda::GpuMat& colorImageR) {

	imageL = colorImageL.clone();
	imageR = colorImageR.clone();
}

cuda::GpuMat NovelViewGenerator::renderLazyNovelView(
	const cuda::GpuMat& srcImage,
	const bool invertT,
	const cuda::GpuMat& warpComposition_x,
	const cuda::GpuMat& warpComposition_y,
	const cuda::GpuMat& shift,
	cuda::Stream stream) {

	cuda::GpuMat novelView;
	cuda::remap(srcImage, novelView, warpComposition_x, warpComposition_y, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(), stream);
	// so far we haven't quite set things up to exactly match the original
	// O(n^3) algorithm. we need to blend the two novel views based on the
	// time shift value. we will pack that into the alpha channel here,
	// then use it to blend the two later.
	alpha_cuda(shift, novelView, stream);
	return novelView;
}

template <typename V, typename T>
inline V lerp(const V x0, const V x1, const T alpha) {
	return x0 * (T(1) - alpha) + x1 * alpha;
}

template <typename T>
inline T lerp(const T x0, const T x1, const T alpha) {
	return x0 * (T(1) - alpha) + x1 * alpha;
}


cuda::GpuMat NovelViewGenerator::combineLazyNovelViews( 
	const cuda::GpuMat& warpComposition_lx, 
	const cuda::GpuMat& warpComposition_ly,
	const cuda::GpuMat& warpComposition_rx,
	const cuda::GpuMat& warpComposition_ry,
	const cuda::GpuMat& flowMag_l,
	const cuda::GpuMat& flowMag_r,
	const cuda::GpuMat& shift_l,
	const cuda::GpuMat& shift_r,
	cuda::Stream streaml,
	cuda::Stream streamr){
	cuda::GpuMat leftEyeFromLeft = renderLazyNovelView(
		imageL,
		false, 
		warpComposition_lx,
		warpComposition_ly,
		shift_l,
		streaml);
	cuda::GpuMat leftEyeFromRight = renderLazyNovelView(
		imageR,
		true,
		warpComposition_rx,
		warpComposition_ry,
		shift_r,
		streamr);
	streaml.waitForCompletion();
	streamr.waitForCompletion();

	cuda::GpuMat leftEyeCombined;
	combine_cuda(leftEyeFromLeft,
		leftEyeFromRight,
		flowMag_l,
		flowMag_r,
		leftEyeCombined,
		streaml);
	streaml.waitForCompletion();
	return leftEyeCombined;

}