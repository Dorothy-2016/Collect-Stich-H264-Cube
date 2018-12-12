#include "NovelViewUpdate.h"

using namespace cv;


static constexpr int kPyrMinImageSize = 24;
static constexpr int kPyrMaxLevels = 1000;
static constexpr float kGradEpsilon = 0.001f; // for finite differences
static constexpr float kUpdateAlphaThreshold = 0.9f;   // pixels with alpha below this aren't updated by proposals
static constexpr int kMedianBlurSize = 5;      // medianBlur max size is 5 pixels for CV_32FC2
static constexpr int kPreBlurKernelWidth = 5;
static constexpr float kPreBlurSigma = 0.25f;  // amount to blur images before pyramids
static constexpr int kFinalFlowBlurKernelWidth = 3;
static constexpr float kFinalFlowBlurSigma = 1.0f;   // blur that is applied to flow at the end after upscaling
static constexpr int kGradientBlurKernelWidth = 3;
static constexpr float kGradientBlurSigma = 0.5f;   // amount to blur image gradients
static constexpr int kBlurredFlowKernelWidth = 15;     // for regularization/smoothing/diffusion
static constexpr float kBlurredFlowSigma = 8.0f;
static constexpr float pyrScaleFactor = 0.9f;

static const float smoothnessCoef = 0.001f;
static const float verticalRegularizationCoef = 0.01f;
static const float horizontalRegularizationCoef = 0.01f;
static const float gradientStepSize = 0.5f;
static const float kDownscaleFactor = 0.5f;
static const float kDirectionalRegularizationCoef = 0.0f;




  vector<Mat> buildPyramid(const Mat& src) {
    vector<Mat> pyramid = {src};
    while (pyramid.size() < kPyrMaxLevels) {
      Size newSize(
        pyramid.back().cols * pyrScaleFactor + 0.5f,
        pyramid.back().rows * pyrScaleFactor + 0.5f);
      if (newSize.height <= kPyrMinImageSize || newSize.width <= kPyrMinImageSize) {
        break;
      }
      Mat scaledImage;
      resize(pyramid.back(), scaledImage, newSize, 0, 0, CV_INTER_LINEAR);
      pyramid.push_back(scaledImage);
    }
    return pyramid;
  }

  static inline float getPixBilinear32FExtend(const Mat& img, float x, float y) {
	  const cv::Size& imgSize = img.size();
	  x = min(imgSize.width - 2.0f, max(0.0f, x));
	  y = min(imgSize.height - 2.0f, max(0.0f, y));
	  const int x0 = int(x);
	  const int y0 = int(y);
	  const float xR = x - float(x0);
	  const float yR = y - float(y0);
	  const float* p = img.ptr<float>(y0);
	  const float f00 = *(p + x0);
	  const float f01 = *(p + x0 + img.cols);
	  const float f10 = *(p + x0 + 1);
	  const float f11 = *(p + x0 + img.cols + 1);
	  const float a1 = f00;
	  const float a2 = f10 - f00;
	  const float a3 = f01 - f00;
	  const float a4 = f00 + f11 - f10 - f01;
	  return a1 + a2 * xR + a3 * yR + a4 * xR * yR;
  }


  inline float errorFunction(
	  const Mat& I0,
	  const Mat& I1,
	  const Mat& alpha0,
	  const Mat& alpha1,
	  const Mat& I0x,
	  const Mat& I0y,
	  const Mat& I1x,
	  const Mat& I1y,
	  const int x,
	  const int y,
	  const Mat& flow,
	  const Mat& blurredFlow,
	  const Point2f& flowDir) {

	  const float matchX = x + flowDir.x;
	  const float matchY = y + flowDir.y;
	  const float i0x = I0x.at<float>(y, x);
	  const float i0y = I0y.at<float>(y, x);
	  const float i1x = getPixBilinear32FExtend(I1x, matchX, matchY);
	  const float i1y = getPixBilinear32FExtend(I1y, matchX, matchY);
	  const Point2f flowDiff = blurredFlow.at<Point2f>(y, x) - flowDir;
	  const float smoothness = sqrtf(flowDiff.dot(flowDiff));

	  float err = sqrtf((i0x - i1x) * (i0x - i1x) + (i0y - i1y) * (i0y - i1y))
		  + smoothness * smoothnessCoef
		  + verticalRegularizationCoef * fabsf(flowDir.y) / float(I0.cols)
		  + horizontalRegularizationCoef * fabsf(flowDir.x) / float(I0.cols);

	  return err;
  }
  inline void proposeFlowUpdate(
	  const Mat& alpha0,
	  const Mat& alpha1,
	  const Mat& I0,
	  const Mat& I1,
	  const Mat& I0x,
	  const Mat& I0y,
	  const Mat& I1x,
	  const Mat& I1y,
	  Mat& flow,
	  const Mat& blurredFlow,
	  float& currErr,
	  const int updateX, const int updateY,
	  const Point2f& proposedFlow) {

	  const float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, updateX, updateY, flow, blurredFlow, proposedFlow);
	  if (proposalErr < currErr) {
		  flow.at<Point2f>(updateY, updateX) = proposedFlow;
		  currErr = proposalErr;
	  }
  }
  inline Point2f errorGradient(
	  const Mat& I0,
	  const Mat& I1,
	  const Mat& alpha0,
	  const Mat& alpha1,
	  const Mat& I0x,
	  const Mat& I0y,
	  const Mat& I1x,
	  const Mat& I1y,
	  const int x,
	  const int y,
	  const Mat& flow,
	  const Mat& blurredFlow,
	  const float currErr) {

	  const static Point2f dx(kGradEpsilon, 0.0f);
	  const static Point2f dy(0.0f, kGradEpsilon);

	  const float fx = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x) + dx);
	  const float fy = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x) + dy);

	  return Point2f((fx - currErr) / kGradEpsilon, (fy - currErr) / kGradEpsilon);
  }
  void lowAlphaFlowDiffusion(const Mat& alpha0, const Mat& alpha1, Mat& flow) {
	  Mat blurredFlow;
	  GaussianBlur(
		  flow,
		  blurredFlow,
		  Size(kBlurredFlowKernelWidth, kBlurredFlowKernelWidth),
		  kBlurredFlowSigma);
	  for (int y = 0; y < flow.rows; ++y) {
		  for (int x = 0; x < flow.cols; ++x) {
			  const float a0 = alpha0.at<float>(y, x);
			  const float a1 = alpha1.at<float>(y, x);
			  const float diffusionCoef = 1.0f - a0 * a1;
			  flow.at<Point2f>(y, x) =
				  diffusionCoef * blurredFlow.at<Point2f>(y, x)
				  + (1.0f - diffusionCoef) * flow.at<Point2f>(y, x);
		  }
	  }
  }

  void patchMatchPropagationAndSearch(
	  const Mat& I0,
	  const Mat& I1,
	  const Mat& alpha0,
	  const Mat& alpha1,
	  Mat& flow) {

	  // image gradients
	  Mat I0x, I0y, I1x, I1y;
	  const int kSameDepth = -1; // same depth as source image
	  const int kKernelSize = 1;
	  Sobel(I0, I0x, kSameDepth, 1, 0, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
	  Sobel(I0, I0y, kSameDepth, 0, 1, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
	  Sobel(I1, I1x, kSameDepth, 1, 0, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
	  Sobel(I1, I1y, kSameDepth, 0, 1, kKernelSize, 1, 0.0f, BORDER_REPLICATE);

	  // blur gradients
	  const cv::Size kGradientBlurSize(kGradientBlurKernelWidth, kGradientBlurKernelWidth);
	  GaussianBlur(I0x, I0x, kGradientBlurSize, kGradientBlurSigma);
	  GaussianBlur(I0y, I0y, kGradientBlurSize, kGradientBlurSigma);
	  GaussianBlur(I1x, I1x, kGradientBlurSize, kGradientBlurSigma);
	  GaussianBlur(I1y, I1y, kGradientBlurSize, kGradientBlurSigma);

	  if (flow.empty()) {
		  // initialize to all zeros
		  flow = Mat::zeros(I0.size(), CV_32FC2);
	  }

	  // blur flow. we will regularize against this
	  Mat blurredFlow;
	  GaussianBlur(
		  flow,
		  blurredFlow,
		  cv::Size(kBlurredFlowKernelWidth, kBlurredFlowKernelWidth),
		  kBlurredFlowSigma);

	  const cv::Size imgSize = I0.size();

	  // sweep from top/left
	  for (int y = 0; y < imgSize.height; ++y) {
		  for (int x = 0; x < imgSize.width; ++x) {
			  if (alpha0.at<float>(y, x) > kUpdateAlphaThreshold && alpha1.at<float>(y, x) > kUpdateAlphaThreshold) {
				  float currErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x));
				  if (x > 0) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y, x - 1)); }
				  if (y > 0) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y - 1, x)); }
				  flow.at<Point2f>(y, x) -= gradientStepSize * errorGradient(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, currErr);
			  }
		  }
	  }
	  medianBlur(flow, flow, kMedianBlurSize);

	  // sweep from bottom/right
	  for (int y = imgSize.height - 1; y >= 0; --y) {
		  for (int x = imgSize.width - 1; x >= 0; --x) {
			  if (alpha0.at<float>(y, x) > kUpdateAlphaThreshold && alpha1.at<float>(y, x) > kUpdateAlphaThreshold) {
				  float currErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x));
				  if (x < imgSize.width - 1) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y, x + 1)); }
				  if (y < imgSize.height - 1) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y + 1, x)); }
				  flow.at<Point2f>(y, x) -= gradientStepSize * errorGradient(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, currErr);
			  }
		  }
	  }
	  medianBlur(flow, flow, kMedianBlurSize);
	  lowAlphaFlowDiffusion(alpha0, alpha1, flow);
  }

  void adjustFlowTowardPrevious(const Mat& prevFlow, const Mat& motion, Mat& flow) {
	  for (int y = 0; y < flow.rows; ++y) {
		  for (int x = 0; x < flow.cols; ++x) {
			  const float w = 1.0f - motion.at<float>(y, x);
			  flow.at<Point2f>(y, x) =
				  flow.at<Point2f>(y, x) * (1.0f - w) + prevFlow.at<Point2f>(y, x) * w;
		  }
	  }
  }

  bool usePrevFlowTemporalRegularization = false;

  void NovelViewUpdate::computeOpticalFlow(
	  const Mat& rgba0byte,
	  const Mat& rgba1byte,
	  const Mat& prevFlow,
	  const Mat& prevI0BGRA,
	  const Mat& prevI1BGRA,
	  Mat& flow) {

	  assert(prevFlow.dims == 0 || prevFlow.size() == rgba0byte.size());

	  // pre-scale everything to a smaller size. this should be faster + more stable
	  Mat rgba0byteDownscaled, rgba1byteDownscaled, prevFlowDownscaled;
	  Mat prevI0BGRADownscaled, prevI1BGRADownscaled;
	  double downscaleFactor = 0.5;
	  cv::Size originalSize = rgba0byte.size();
	  cv::Size downscaleSize(
		  rgba0byte.cols * downscaleFactor, rgba0byte.rows * downscaleFactor);
	  resize(rgba0byte, rgba0byteDownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
	  resize(rgba1byte, rgba1byteDownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
	  Mat motion = Mat(downscaleSize, CV_32F);
	  if (prevFlow.dims > 0) {
		  usePrevFlowTemporalRegularization = true;
		  resize(prevFlow, prevFlowDownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
		  prevFlowDownscaled *= float(prevFlowDownscaled.rows) / float(prevFlow.rows);

		  resize(prevI0BGRA, prevI0BGRADownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
		  resize(prevI1BGRA, prevI1BGRADownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);

		  // do motion detection vs. previous frame's images
		  for (int y = 0; y < rgba0byteDownscaled.rows; ++y) {
			  for (int x = 0; x < rgba0byteDownscaled.cols; ++x) {
				  motion.at<float>(y, x) =
					  (fabs(rgba1byteDownscaled.at<Vec4b>(y, x)[0] - prevI1BGRADownscaled.at<Vec4b>(y, x)[0])*1.5f +
						  fabs(rgba1byteDownscaled.at<Vec4b>(y, x)[1] - prevI1BGRADownscaled.at<Vec4b>(y, x)[1])*1.5f +
						  fabs(rgba1byteDownscaled.at<Vec4b>(y, x)[2] - prevI1BGRADownscaled.at<Vec4b>(y, x)[2])*1.5f) / (255.0f * 3.0f);
			  }
		  }
	  }
	  // convert to various color spaces
	  Mat I0Grey, I1Grey, I0, I1, alpha0, alpha1;
	  Mat channels0[4], channels1[4];
	  split(rgba0byteDownscaled, channels0);
	  split(rgba1byteDownscaled, channels1);
	  cvtColor(rgba0byteDownscaled, I0Grey, CV_BGRA2GRAY);
	  cvtColor(rgba1byteDownscaled, I1Grey, CV_BGRA2GRAY);
	  I0Grey.convertTo(I0, CV_32F);
	  I1Grey.convertTo(I1, CV_32F);
	  I0 /= 255.0f;
	  I1 /= 255.0f;
	  channels0[3].convertTo(alpha0, CV_32F);
	  channels1[3].convertTo(alpha1, CV_32F);
	  alpha0 /= 255.0f;
	  alpha1 /= 255.0f;

	  GaussianBlur(I0, I0, Size(5, 5), 0.25f);
	  GaussianBlur(I1, I1, Size(5, 5), 0.25f);

	  vector<Mat> pyramidI0 = buildPyramid(I0);
	  vector<Mat> pyramidI1 = buildPyramid(I1);
	  vector<Mat> pyramidAlpha0 = buildPyramid(alpha0);
	  vector<Mat> pyramidAlpha1 = buildPyramid(alpha1);
	  vector<Mat> prevFlowPyramid = buildPyramid(prevFlowDownscaled);
	  vector<Mat> motionPyramid = buildPyramid(motion);

	  flow = Mat();

	  for (int level = pyramidI0.size() - 1; level >= 0; --level) {
		  patchMatchPropagationAndSearch(
			  pyramidI0[level],
			  pyramidI1[level],
			  pyramidAlpha0[level],
			  pyramidAlpha1[level],
			  flow);

		  if (usePrevFlowTemporalRegularization) {
			  adjustFlowTowardPrevious(prevFlowPyramid[level], motionPyramid[level], flow);
		  }

		  if (level > 0) { // scale the flow up to the next size
			  resize(flow, flow, pyramidI0[level - 1].size(), 0, 0, CV_INTER_CUBIC);
			  flow *= (1.0f / pyrScaleFactor);
		  }
	  }

	  // scale the flow result back to full size
	  resize(flow, flow, originalSize, 0, 0, CV_INTER_LINEAR);
	  flow *= (1.0f / downscaleFactor);
	  GaussianBlur(
		  flow,
		  flow,
		  Size(kFinalFlowBlurKernelWidth, kFinalFlowBlurKernelWidth),
		  kFinalFlowBlurSigma);
  }


void NovelViewUpdate::prepare(
	const cv::Mat& colorImageL,
	const cv::Mat& colorImageR,
	Mat* prevFlowLtoR,
	Mat* prevFlowRtoL,
	Mat* prevColorImageL,
	Mat* prevColorImageR) {

	imageL = colorImageL.clone();
	imageR = colorImageR.clone();
	Mat tmpL, tmpR;
	Mat frame0, frame1;


	computeOpticalFlow(imageL, imageR, *prevFlowLtoR, *prevColorImageL, *prevColorImageR, tmpL);
	computeOpticalFlow(imageR, imageL, *prevFlowRtoL, *prevColorImageR, *prevColorImageL, tmpR);

	flowLtoR = tmpL.clone();
	flowRtoL = tmpR.clone();

	*prevFlowLtoR = tmpL.clone();
	*prevFlowRtoL = tmpR.clone();

	*prevColorImageL = colorImageL.clone();
	*prevColorImageR = colorImageR.clone();

}


void NovelViewUpdate::renderLazyNovelView(
	const int width,
	const int height,
	const vector<vector<cv::Point3f>>& novelViewWarpBuffer,
	const cv::Mat& srcImage,
	const cv::Mat& opticalFlow,
	const bool invertT,
	cv::Mat* warp_x,
	cv::Mat* warp_y,
	cv::Mat* flowMag,
	cv::Mat* shiftT) {

	// a composition of remap
	Mat warpOpticalFlow_x = Mat(Size(width, height), CV_32F);
	Mat warpOpticalFlow_y = Mat(Size(width, height), CV_32F);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const Point3f lazyWarp = novelViewWarpBuffer[x][y];
			warpOpticalFlow_x.at<float>(y, x) = lazyWarp.x;
			warpOpticalFlow_y.at<float>(y, x) = lazyWarp.y;
		}
	}
	Mat remappedFlow;
	remap(opticalFlow, remappedFlow, warpOpticalFlow_x, warpOpticalFlow_y, CV_INTER_CUBIC);

	Mat warpComposition_x = Mat(Size(width, height), CV_32F);
	Mat warpComposition_y = Mat(Size(width, height), CV_32F);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const Point3f lazyWarp = novelViewWarpBuffer[x][y];
			Point2f flowDir = remappedFlow.at<Point2f>(y, x);
			// the 3rd coordinate (z) of novelViewWarpBuffer is shift/time value
			const float t = invertT ? (1.0f - lazyWarp.z) : lazyWarp.z;
			warpComposition_x.at<float>(y, x) = lazyWarp.x + flowDir.x * t;
			warpComposition_y.at<float>(y, x) = lazyWarp.y + flowDir.y * t;
		}
	}

	*warp_x = warpComposition_x.clone();
	*warp_y = warpComposition_y.clone();

	Mat novelView;

	remap(srcImage, novelView, warpComposition_x, warpComposition_y, CV_INTER_CUBIC);
	Mat novelViewFlowMag(novelView.size(), CV_32F);
	Mat novelViewT(novelView.size(), CV_32F);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			const Point3f lazyWarp = novelViewWarpBuffer[x][y];
			Point2f flowDir = remappedFlow.at<Point2f>(y, x);
			const float t = invertT ? (1.0f - lazyWarp.z) : lazyWarp.z;
			novelView.at<Vec4b>(y, x)[3] =
				int((1.0f - t) * novelView.at<Vec4b>(y, x)[3]);
			novelViewT.at<float>(y, x) = 1.0f - t;
			novelViewFlowMag.at<float>(y, x) =
				sqrtf(flowDir.x * flowDir.x + flowDir.y * flowDir.y);
		}
	}

	
	*flowMag = novelViewFlowMag.clone();
	*shiftT = novelViewT.clone();
}

template <typename V, typename T>
inline V lerp(const V x0, const V x1, const T alpha) {
	return x0 * (T(1) - alpha) + x1 * alpha;
}

template <typename T>
inline T lerp(const T x0, const T x1, const T alpha) {
	return x0 * (T(1) - alpha) + x1 * alpha;
}


void NovelViewUpdate::combineLazyNovelViews(
	const LazyNovelViewBuffer& lazyBuffer,
	cv::Mat* warpComposition_lx,
	cv::Mat* warpComposition_ly,
	cv::Mat* warpComposition_rx,
	cv::Mat* warpComposition_ry,
	cv::Mat* flowMag_l,
	cv::Mat* flowMag_r,
	cv::Mat* shift_l,
	cv::Mat* shift_r){


	renderLazyNovelView(
		lazyBuffer.width, lazyBuffer.height,
		lazyBuffer.warpL,
		imageL,
		flowRtoL,
		false,
		warpComposition_lx,
		warpComposition_ly,
		flowMag_l,
		shift_l);
	renderLazyNovelView(
		lazyBuffer.width, lazyBuffer.height,
		lazyBuffer.warpL,
		imageR,
		flowLtoR,
		true,
		warpComposition_rx,
		warpComposition_ry,
		flowMag_r,
		shift_r);

}
