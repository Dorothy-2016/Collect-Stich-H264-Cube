#include <fstream>
#include <iostream>
#include <chrono>
#include "RigDescription.h"
#include "NovelViewGenerator.h"
#include "NovelViewUpdate.h"
#include "readwrite.h"
#include "EncVideo.h"
using namespace surround360;
using namespace std::chrono;
using namespace std;
using namespace cv;
#pragma comment(lib,"ws2_32.lib")

//#define Debug
bool flag_send = true;

#define FLAGS_eqr_height 1024
#define FLAGS_eqr_width  2046
#define FLAGS_final_width 2048
#define FLAGS_side_cam_num 6
#define FLAGS_std_alpha_feather_size 31

cuda::Stream streamL[FLAGS_side_cam_num];
cuda::Stream streamR[FLAGS_side_cam_num];
vector<cuda::GpuMat> projection_x, projection_y, composition_lx, composition_ly, composition_rx, composition_ry, 
flowMag_l, flowMag_r, shift_l, shift_r;

cuda::Stream streamTS;


Mat firstpro_x[FLAGS_side_cam_num], firstpro_y[FLAGS_side_cam_num], firstpos_lx[FLAGS_side_cam_num], firstpos_ly[FLAGS_side_cam_num], firstpos_rx[FLAGS_side_cam_num], firstpos_ry[FLAGS_side_cam_num],
firstmag_l[FLAGS_side_cam_num], firstmag_r[FLAGS_side_cam_num], firstshi_l[FLAGS_side_cam_num], firstshi_r[FLAGS_side_cam_num];

Mat topLR_x, topLR_y, topSide_x, topSide_y, topRamp;


VideoCapture CAM[FLAGS_side_cam_num+1];


void feather_alpha_chanel(
	const cuda::GpuMat& src,
	cuda::GpuMat& dst,
	const int feathersize,
	cuda::Stream& stream = cuda::Stream::Null());

void Gpu_strip(
	const cuda::GpuMat& fisheyeSpherical,
	const cuda::GpuMat& warpedExtendedFisheyeSpherical,
	const cuda::GpuMat& warpedSphericalForEye,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null());

void extendimage(const cuda::GpuMat& gpucroppedSideSpherical,
	cuda::GpuMat& gpuextendedSideSpherical,
	const cuda::GpuMat& gpufisheyeSpherical,
	cuda::GpuMat& gpuextendedFisheyeSpherical,
	const int extendedWidth,
	cuda::Stream& stream = cuda::Stream::Null());

void Gpu_flatten(
	const cuda::GpuMat& bottomLayer,
	const cuda::GpuMat& topLayer,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null());

void alpha_cuda(const cuda::GpuMat& shiftMat,
	cuda::GpuMat& dst,
	cuda::Stream& stream = cuda::Stream::Null());

void camInit(){
	CAM[0] = VideoCapture(6);
	CAM[1] = VideoCapture(5);
	CAM[2] = VideoCapture(1);
	CAM[3] = VideoCapture(4);
	CAM[4] = VideoCapture(3);
	CAM[5] = VideoCapture(2);
	CAM[6] = VideoCapture(0);
	for (int i = 0; i< FLAGS_side_cam_num+1; i++ ){
		//CAM[i] = VideoCapture(i);
		if (!CAM[i].isOpened()) {
			cout << "cam" << i << " error" << endl;
			exit(-1);
		}
		CAM[i].set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
		CAM[i].set(CV_CAP_PROP_FRAME_WIDTH, 1280);//宽度 
		CAM[i].set(CV_CAP_PROP_FRAME_HEIGHT, 720);//高度
		CAM[i].set(CV_CAP_PROP_BRIGHTNESS, 0);//亮度 1
		CAM[i].set(CV_CAP_PROP_CONTRAST, 32);//对比度 40p
		CAM[i].set(CV_CAP_PROP_SATURATION, 65);//饱和度 50
		CAM[i].set(CV_CAP_PROP_HUE, 0);//色调 50
		CAM[i].set(CV_CAP_PROP_EXPOSURE, -5);//曝光 50
	}
}


void readInitMat() {
	string projectpath = "Mat/Side/Projection/";
	string compositionpath = "Mat/Side/WarpComposition/";
	string flowmagpath = "Mat/Side/FlowMag/";
	string shiftpath = "Mat/Side/Shift/";
	for (int i = 0; i<FLAGS_side_cam_num; i++) {
		Mat warp_x, warp_y;
		cuda::GpuMat gpuwarp_x, gpuwarp_y;
		//warp_x = readMatFromFile(projectpath + "warp_x_" + to_string(i) + ".xml");
		//warp_y = readMatFromFile(projectpath + "warp_y_" + to_string(i) + ".xml");
		warp_x = firstpro_x[i];
		warp_y = firstpro_y[i];
		gpuwarp_x.upload(warp_x);
		gpuwarp_y.upload(warp_y);
		projection_x.push_back(gpuwarp_x);
		projection_y.push_back(gpuwarp_y);

		Mat warpComposition_lx, warpComposition_ly, warpComposition_rx, warpComposition_ry;
		cuda::GpuMat gwarpComposition_lx, gwarpComposition_ly, gwarpComposition_rx, gwarpComposition_ry;
		//warpComposition_lx = readMatFromFile(compositionpath + "warpComposition_lx_" + to_string(i) + ".xml");
		//warpComposition_ly = readMatFromFile(compositionpath + "warpComposition_ly_" + to_string(i) + ".xml");
		//warpComposition_rx = readMatFromFile(compositionpath + "warpComposition_rx_" + to_string(i) + ".xml");
		//warpComposition_ry = readMatFromFile(compositionpath + "warpComposition_ry_" + to_string(i) + ".xml");

		warpComposition_lx = firstpos_lx[i];
		warpComposition_ly = firstpos_ly[i];
		warpComposition_rx = firstpos_rx[i];
		warpComposition_ry = firstpos_ry[i];

		gwarpComposition_lx.upload(warpComposition_lx);
		gwarpComposition_ly.upload(warpComposition_ly);
		gwarpComposition_rx.upload(warpComposition_rx);
		gwarpComposition_ry.upload(warpComposition_ry);
		composition_lx.push_back(gwarpComposition_lx);
		composition_ly.push_back(gwarpComposition_ly);
		composition_rx.push_back(gwarpComposition_rx);
		composition_ry.push_back(gwarpComposition_ry);

		Mat flowmag_l, flowmag_r;
		cuda::GpuMat gflowmag_l, gflowmag_r;
		//flowmag_l = readMatFromFile(flowmagpath + "novelViewFlowMag_l_" + to_string(i) + ".xml");
		//flowmag_r = readMatFromFile(flowmagpath + "novelViewFlowMag_r_" + to_string(i) + ".xml");
		flowmag_l = firstmag_l[i];
		flowmag_r = firstmag_r[i];
		gflowmag_l.upload(flowmag_l);
		gflowmag_r.upload(flowmag_r);
		flowMag_l.push_back(gflowmag_l);
		flowMag_r.push_back(gflowmag_r);

		Mat t_l, t_r;
		cuda::GpuMat gt_l, gt_r;
		//t_l = readMatFromFile(shiftpath + "novelViewT_l_" + to_string(i) + ".xml");
		//t_r = readMatFromFile(shiftpath + "novelViewT_r_" + to_string(i) + ".xml");
		t_l = firstshi_l[i];
		t_r = firstshi_r[i];
		gt_l.upload(t_l);
		gt_r.upload(t_r);
		shift_l.push_back(gt_l);
		shift_r.push_back(gt_r);
	}
}

static double getCurrTimeSec() {
	return (double)(system_clock::now().time_since_epoch().count()) * system_clock::period::num / system_clock::period::den;
}

float approximateFov(const Camera& camera, const bool vertical) {
	Camera::Vector2 a = camera.principal;
	Camera::Vector2 b = camera.principal;
	if (vertical) {
		a.y() = 0;
		b.y() = camera.resolution.y();
	}
	else {
		a.x() = 0;
		b.x() = camera.resolution.x();
	}
	return acos(max(
		camera.rig(a).direction().dot(camera.forward()),
		camera.rig(b).direction().dot(camera.forward())));
}

float approximateFov(const Camera::Rig& rig, const bool vertical) {
	float result = 0;
	for (const auto& camera : rig) {
		result = max(result, approximateFov(camera, vertical));
	}
	return result;
}

static inline float toRadians(const float deg) { return deg * M_PI / 180.0f; }
static inline double toRadians(const double deg) { return deg * M_PI / 180.0; }

void bicubicRemapToSpherical(
	cuda::GpuMat& dst,
	const cuda::GpuMat& src,
	const Camera& camera,
	const float leftAngle,
	const float rightAngle,
	const float topAngle,
	const float bottomAngle,
	const int idx,
	cuda::Stream stream) {

	cuda::GpuMat tmp = src;
	if (src.channels() == 3 && dst.channels() == 4) {
		cuda::cvtColor(src, tmp, CV_BGR2BGRA, 4, stream);
	}
	
}

void projectSideToSpherical(
	cuda::GpuMat& dst,
	const cuda::GpuMat& src,
	const Camera& camera,
	const float leftAngle,
	const float rightAngle,
	const float topAngle,
	const float bottomAngle,
	const int idx) {

	// convert, clone or reference, as needed
	cuda::GpuMat tmp = src;
	if (src.channels() == 3) {
		cuda::cvtColor(src, tmp, CV_BGR2BGRA, 4, streamL[idx]);
	}

	cuda::remap(tmp, dst, projection_x[idx], projection_y[idx], CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(), streamL[idx]);

}

void projectSphericalCamImages(
	const RigDescription& rig,
	const string& imagesDir,
	const string& frameNumber,
	vector<cuda::GpuMat>& projectionImages) {

	const double startload = getCurrTimeSec();
	Mat frame[FLAGS_side_cam_num];
	for (int i = 0; i < FLAGS_side_cam_num; i++)
		CAM[i].read(frame[i]);
	vector<cuda::GpuMat> camImages = rig.loadSideCameraImages(frame, streamL);
	const double endload = getCurrTimeSec();
	cout << "load time :" << endload - startload << endl;
	projectionImages.resize(camImages.size());
	vector<std::thread> threads;
	const float hRadians = 2 * approximateFov(rig.rigSideOnly, false);
	const float vRadians = 2 * approximateFov(rig.rigSideOnly, true);

	for (int camIdx = 0; camIdx < camImages.size(); ++camIdx) {
		const Camera& camera = rig.rigSideOnly[camIdx];
		projectionImages[camIdx].create(
			FLAGS_eqr_height * vRadians / M_PI,
			FLAGS_eqr_width * hRadians / (2 * M_PI),
			CV_8UC4);
		// the negative sign here is so the camera array goes clockwise
		float direction = -float(camIdx) / float(camImages.size()) * 2.0f * M_PI;
		threads.emplace_back(
			projectSideToSpherical,
			ref(projectionImages[camIdx]),
			cref(camImages[camIdx]),
			cref(camera),
			direction + hRadians / 2,
			direction - hRadians / 2,
			vRadians / 2,
			-vRadians / 2,
			camIdx);
	}
	for (std::thread& t : threads) { t.join(); }
	for(int camIdx = 0; camIdx < camImages.size(); ++camIdx)
		streamL[camIdx].waitForCompletion();

	cout << "projectione time:" << getCurrTimeSec() - endload << endl;
}

void prepareNovelViewGeneratorThread(
	const int overlapImageWidth,
	const int leftIdx, // only used to determine debug image filename
	cuda::GpuMat* imageL,
	cuda::GpuMat* imageR,
	NovelViewGenerator* novelViewGen) {

	cuda::GpuMat overlapImageL = (*imageL)(Rect(
		imageL->cols - overlapImageWidth, 0, overlapImageWidth, imageL->rows));
	cuda::GpuMat overlapImageR = (*imageR)(Rect(0, 0, overlapImageWidth, imageR->rows));

	// this is the call to actually compute optical flow
	novelViewGen->prepare(
		overlapImageL,
		overlapImageR);

}

cuda::GpuMat stackHorizontal(const std::vector<cuda::GpuMat>& images, const int width, const int height) {

	cuda::GpuMat stacked;
	stacked.create(Size(FLAGS_eqr_width, height), images[0].type());
	for (int i = 0; i<FLAGS_side_cam_num; i++) {
		cuda::GpuMat tmp = stacked.colRange(width * i, width * (i + 1));
		images[i].copyTo(tmp);
	}
	return stacked;
}

Mat stackHorizontal(const std::vector<Mat>& images) {
	assert(!images.empty());
	if (images.size() == 1) {
		return images[0];
	}
	Mat stacked = images[0].clone();
	for (int i = 1; i < images.size(); ++i) {
		hconcat(stacked, images[i], stacked);
	}
	return stacked;
}

void renderStereoPanoramaChunksThread(
	const int leftIdx, // left camera
	const int numCams,
	const int camImageWidth,
	const int camImageHeight,
	const int numNovelViews,
	const float fovHorizontalRadians,
	NovelViewGenerator* novelViewGen,
	cuda::GpuMat* chunkL) {

	const int rightIdx = (leftIdx + 1) % numCams;
	cuda::GpuMat lazyNovelChunksLR =
		novelViewGen->combineLazyNovelViews(
			composition_lx[leftIdx], composition_ly[leftIdx], 
			composition_rx[leftIdx], composition_ry[leftIdx],
			flowMag_l[leftIdx], flowMag_r[leftIdx],
			shift_l[leftIdx], shift_r[leftIdx],
			streamL[leftIdx],
			streamR[leftIdx]);


	*chunkL = lazyNovelChunksLR;
}

void generateRingOfNovelViewsAndRenderStereoSpherical(
	const float cameraRingRadius,
	const float camFovHorizontalDegrees,
	vector<cuda::GpuMat>& projectionImages,
	cuda::GpuMat& panoImageL,
	cuda::GpuMat& panoImageR,
	double& opticalFlowRuntime,
	double& novelViewRuntime) {

	const int numCams = projectionImages.size();

	const float fovHorizontalRadians = toRadians(camFovHorizontalDegrees);
	const float overlapAngleDegrees =
		(camFovHorizontalDegrees * float(numCams) - 360.0) / float(numCams);
	const int camImageWidth = projectionImages[0].cols;
	const int camImageHeight = projectionImages[0].rows;
	const int overlapImageWidth =
		float(camImageWidth) * (overlapAngleDegrees / camFovHorizontalDegrees);
	const int numNovelViews = camImageWidth - overlapImageWidth; // per image pair

																 // setup parallel optical flow
	double startOpticalFlowTime = getCurrTimeSec();
	vector<NovelViewGenerator*> novelViewGenerators(projectionImages.size());
	vector<std::thread> threads;
	for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
		const int rightIdx = (leftIdx + 1) % projectionImages.size();
		novelViewGenerators[leftIdx] = new NovelViewGenerator();
		threads.push_back(std::thread(
			prepareNovelViewGeneratorThread,
			overlapImageWidth,
			leftIdx,
			&projectionImages[leftIdx],
			&projectionImages[rightIdx],
			novelViewGenerators[leftIdx]
		));
	}
	for (std::thread& t : threads) { t.join(); }

	vector<cuda::GpuMat> panoChunks(FLAGS_side_cam_num);

	vector<std::thread> panoThreads;
	for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
		panoThreads.push_back(std::thread(
			renderStereoPanoramaChunksThread,
			leftIdx,
			numCams,
			camImageWidth,
			camImageHeight,
			numNovelViews,
			fovHorizontalRadians,
			novelViewGenerators[leftIdx],
			&panoChunks[leftIdx]
		));
	}
	for (std::thread& t : panoThreads) { t.join(); }

	for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
		delete novelViewGenerators[leftIdx];
	}

	panoImageL = stackHorizontal(panoChunks, FLAGS_eqr_width / FLAGS_side_cam_num, camImageHeight);

	cout << "stitch time:" << getCurrTimeSec() - startOpticalFlowTime << endl;
}

void padToheight(cuda::GpuMat& unpaddedImage, const int targetHeight) {
	const int paddingAbove = (targetHeight - unpaddedImage.rows) / 2;
	const int paddingBelow = targetHeight - unpaddedImage.rows - paddingAbove;
	cuda::copyMakeBorder(
		unpaddedImage,
		unpaddedImage,
		paddingAbove,
		paddingBelow,
		0,
		0,
		BORDER_CONSTANT,
		Scalar(0.0, 0.0, 0.0, 0.0));
}

void bicubicRemapToSpherical(
	Mat& dst,
	Mat& pro_x,
	Mat& pro_y,
	const Mat& src,
	const Camera& camera,
	const float leftAngle,
	const float rightAngle,
	const float topAngle,
	const float bottomAngle) {
	Mat warp_x(dst.size(), CV_32F);
	Mat warp_y(dst.size(), CV_32F);
	for (int x = 0; x < warp_x.cols; ++x) {
		// sweep xAngle from leftAngle to rightAngle
		const float xFrac = (x + 0.5f) / warp_x.cols;
		const float xAngle = (1 - xFrac) * leftAngle + xFrac * rightAngle;
		for (int y = 0; y < warp_x.rows; ++y) {
			// sweep yAngle from topAngle to bottomAngle
			const float yFrac = (y + 0.5f) / warp_x.rows;
			float yAngle = (1 - yFrac) * topAngle + yFrac * bottomAngle;
			const Camera::Vector3 unit(
				cos(yAngle) * cos(xAngle),
				cos(yAngle) * sin(xAngle),
				sin(yAngle));
			const Camera::Vector2 pixel =
				camera.pixel(unit * int(Camera::kNearInfinity));
			warp_x.at<float>(y, x) = pixel.x() - 0.5;
			warp_y.at<float>(y, x) = pixel.y() - 0.5;
		}
	}
	Mat tmp = src;
	if (src.channels() == 3 && dst.channels() == 4) {
		cvtColor(src, tmp, CV_BGR2BGRA);
	}

	pro_x = warp_x.clone();
	pro_y = warp_y.clone();

	remap(tmp, dst, warp_x, warp_y, CV_INTER_CUBIC, BORDER_CONSTANT);
}

void firstProjectSideToSpherical(
	Mat& dst,
	const Mat& src,
	Mat& pro_x,
	Mat& pro_y,
	const Camera& camera,
	const float leftAngle,
	const float rightAngle,
	const float topAngle,
	const float bottomAngle) {

	// convert, clone or reference, as needed
	Mat tmp = src;
	if (src.channels() == 3) {
		cvtColor(src, tmp, CV_BGR2BGRA);
	}

	// remap
	bicubicRemapToSpherical(
		dst,
		pro_x,
		pro_y,
		tmp,
		camera,
		leftAngle,
		rightAngle,
		topAngle,
		bottomAngle);
}


void firstProjectSphericalCamImages(
	const RigDescription& rig,
	const string& imagesDir,
	const string& frameNumber,
	vector<Mat>& projectionImages) {


	vector<Mat> camImages;

	Mat tmp;
	for (int j = 0; j < 5; j++)
		for (int i = 0; i < FLAGS_side_cam_num+1; i++)
			CAM[i].read(tmp);

	for (int i = 0; i < FLAGS_side_cam_num; i++) {
		Mat frame;
		CAM[i].read(frame);
		camImages.push_back(frame);
	}

	projectionImages.resize(camImages.size());

	vector<std::thread> threads;
	const float hRadians = 2 * approximateFov(rig.rigSideOnly, false);
	const float vRadians = 2 * approximateFov(rig.rigSideOnly, true);
	for (int camIdx = 0; camIdx < camImages.size(); ++camIdx) {
		const Camera& camera = rig.rigSideOnly[camIdx];
		projectionImages[camIdx].create(
			FLAGS_eqr_height * vRadians / M_PI,
			FLAGS_eqr_width * hRadians / (2 * M_PI),
			CV_8UC4);
		// the negative sign here is so the camera array goes clockwise
		float direction = -float(camIdx) / float(camImages.size()) * 2.0f * M_PI;
		threads.emplace_back(
			firstProjectSideToSpherical,
			ref(projectionImages[camIdx]),
			cref(camImages[camIdx]),
			ref(firstpro_x[camIdx]),
			ref(firstpro_y[camIdx]),
			cref(camera),
			direction + hRadians / 2,
			direction - hRadians / 2,
			vRadians / 2,
			-vRadians / 2);
	}
	for (std::thread& t : threads) { t.join(); }

}

void firstPrepareThread(
	const int overlapImageWidth,
	const int leftIdx, // only used to determine debug image filename
	Mat* imageL,
	Mat* imageR,
	NovelViewUpdate* novelViewGen) {

	Mat overlapImageL = (*imageL)(Rect(
		imageL->cols - overlapImageWidth, 0, overlapImageWidth, imageL->rows));
	Mat overlapImageR = (*imageR)(Rect(0, 0, overlapImageWidth, imageR->rows));

	novelViewGen->prepare(overlapImageL, overlapImageR, &Mat(), &Mat(), &Mat(), &Mat());

}


void firstRenderThread(
	const int leftIdx, // left camera
	const int numCams,
	const int camImageWidth,
	const int camImageHeight,
	const int numNovelViews,
	const float fovHorizontalRadians,
	NovelViewUpdate* novelViewGen) {


	int currChunkX = 0; // current column in chunk to write
	LazyNovelViewBuffer lazyNovelViewBuffer(FLAGS_eqr_width / numCams, camImageHeight);
	for (int nvIdx = 0; nvIdx < numNovelViews; ++nvIdx) {
		const float shift = float(nvIdx) / float(numNovelViews);
		const float slabShift =
			float(camImageWidth) * 0.5f - float(numNovelViews - nvIdx);

		for (int v = 0; v < camImageHeight; ++v) {
			lazyNovelViewBuffer.warpL[currChunkX][v] =
				Point3f(slabShift, v, shift);
		}
		++currChunkX;
	}

	const int rightIdx = (leftIdx + 1) % numCams;
	novelViewGen->combineLazyNovelViews(
		lazyNovelViewBuffer,
		&firstpos_lx[leftIdx], &firstpos_ly[leftIdx],
		&firstpos_rx[leftIdx], &firstpos_ry[leftIdx],
		&firstmag_l[leftIdx], &firstmag_r[leftIdx],
		&firstshi_l[leftIdx], &firstshi_r[leftIdx]);

}

void firstGenerateRingOfNovelViewsAndRenderStereoSpherical(
	const float cameraRingRadius,
	const float camFovHorizontalDegrees,
	vector<Mat>& projectionImages) {

	const int numCams = projectionImages.size();
	const float fovHorizontalRadians = toRadians(camFovHorizontalDegrees);
	const float overlapAngleDegrees =
		(camFovHorizontalDegrees * float(numCams) - 360.0) / float(numCams);
	const int camImageWidth = projectionImages[0].cols;
	const int camImageHeight = projectionImages[0].rows;
	const int overlapImageWidth =
		float(camImageWidth) * (overlapAngleDegrees / camFovHorizontalDegrees);
	const int numNovelViews = camImageWidth - overlapImageWidth; // per image pair

	vector<NovelViewUpdate*> novelViewGenerators(projectionImages.size());
	vector<std::thread> threads;
	for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
		const int rightIdx = (leftIdx + 1) % projectionImages.size();
		novelViewGenerators[leftIdx] = new NovelViewUpdate();
		threads.push_back(std::thread(
			firstPrepareThread,
			overlapImageWidth,
			leftIdx,
			&projectionImages[leftIdx],
			&projectionImages[rightIdx],
			novelViewGenerators[leftIdx]
		));
	}
	for (std::thread& t : threads) { t.join(); }
	vector<std::thread> panoThreads;
	for (int leftIdx = 0; leftIdx < projectionImages.size(); ++leftIdx) {
		panoThreads.push_back(std::thread(
			firstRenderThread,
			leftIdx,
			numCams,
			camImageWidth,
			camImageHeight,
			numNovelViews,
			fovHorizontalRadians,
			novelViewGenerators[leftIdx]
		));
	}
	for (std::thread& t : panoThreads) { t.join(); }

}

void firstCompute(RigDescription rig) {
	vector<Mat> projectionImages;
	firstProjectSphericalCamImages(rig, "rgb", "000000", projectionImages);

	double opticalFlowRuntime, novelViewRuntime;
	Mat sphericalImageL, sphericalImageR;
	const double fovHorizontal =
		2 * approximateFov(rig.rigSideOnly, false) * (180 / M_PI);

	firstGenerateRingOfNovelViewsAndRenderStereoSpherical(
		rig.getRingRadius(),
		fovHorizontal,
		projectionImages);
}

void ready_for_topside(){
	//读图片

	//导入remap重映射矩阵
	const string matDir = "./Mat/Top";
	topLR_x = readMatFromFile(matDir + "/bicubic_top_project_x.xml");
	topLR_y = readMatFromFile(matDir + "/bicubic_top_project_y.xml");

	topSide_x = readMatFromFile(matDir + "/left_flow_x.xml");
	topSide_y = readMatFromFile(matDir + "/left_flow_y.xml");
}

void ready_for_cuda(
	cuda::GpuMat* gtopImage,
	cuda::GpuMat* gtopLR_x,
	cuda::GpuMat* gtopLR_y,
	cuda::GpuMat* gtopside_x,
	cuda::GpuMat* gtopside_y,
	cuda::GpuMat* gtopramp){
	cuda::Stream stream;
	gtopLR_x->upload(topLR_x, stream);
	gtopLR_y->upload(topLR_y, stream);
	gtopside_x->upload(topSide_x, stream);
	gtopside_y->upload(topSide_y, stream);
	gtopramp->upload(topRamp, stream);

	stream.waitForCompletion();
}

void prepareTopImages(
	const cuda::GpuMat gtopImage,
	const cuda::GpuMat gtopLR_x,
	const cuda::GpuMat gtopLR_y,
	cuda::GpuMat* gtopSpherical) {
	cuda::GpuMat gtopSphericaltemp;
	cuda::remap(gtopImage, gtopSphericaltemp, gtopLR_x, gtopLR_y, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar());
	//gputopSpherical.download(*topSpherical);

	// alpha feather the top spherical image for flow purposes
	cuda::cvtColor(gtopSphericaltemp, gtopSphericaltemp, CV_BGR2BGRA);

	cuda::GpuMat topSphericaldst;

	feather_alpha_chanel(gtopSphericaltemp, *gtopSpherical, FLAGS_std_alpha_feather_size, streamTS);
}

void poleToSideFlow(
	const cuda::GpuMat gtopside_x,
	const cuda::GpuMat gtopside_y,
	const cuda::GpuMat gpuSideSpherical,
	const cuda::GpuMat gpucroppedSideSpherical,
	const cuda::GpuMat gpufisheyeSpherical,
	cuda::GpuMat* gpuextendedSideSpherical,
	cuda::GpuMat* gpuextendedFisheyeSpherical,
	cuda::GpuMat* gpuwarpedExtendedFisheyeSpherical,
	cuda::GpuMat* warpedSphericalForEye,
	const int extendedWidth) {
//	cout << gpucroppedSideSpherical.type() << gpufisheyeSpherical.type() << endl;
	//cuda::GpuMat g1, g2;
	extendimage(
		gpucroppedSideSpherical,
		*gpuextendedSideSpherical,
		gpufisheyeSpherical,
		*gpuextendedFisheyeSpherical,
		extendedWidth,
		streamTS);

	//cuda_src.upload(src,stream_up);
	cuda::remap(*gpuextendedFisheyeSpherical, *gpuwarpedExtendedFisheyeSpherical, gtopside_x, gtopside_y, CV_INTER_CUBIC, BORDER_CONSTANT, Scalar(), streamTS);


	// take the extra strip on the right side and alpha-blend it out on the left side of the result
	cuda::GpuMat warpedSphericalForEyetmp = (*gpuwarpedExtendedFisheyeSpherical)(Rect(0, 0, gpufisheyeSpherical.cols, gpufisheyeSpherical.rows));

	Gpu_strip(gpufisheyeSpherical, *gpuwarpedExtendedFisheyeSpherical, warpedSphericalForEyetmp, *warpedSphericalForEye, streamTS);

	cuda::copyMakeBorder(
		*warpedSphericalForEye,
		*warpedSphericalForEye,
		0,
		gpuSideSpherical.rows - warpedSphericalForEye->rows,
		0,
		0,
		BORDER_CONSTANT,
		Scalar(0, 0, 0, 0));
}

const Camera::Vector3 kGlobalUp = Camera::Vector3::UnitZ();

void prepareTopImagesThread(
	const RigDescription& rig,
	const Mat& topImage,
	Mat* topSpherical) {
	const Camera& camera = rig.findCameraByDirection(kGlobalUp);
	topSpherical->create(
		FLAGS_eqr_height * camera.getFov() / M_PI,
		FLAGS_eqr_width,
		CV_8UC3);

	bicubicRemapToSpherical(
		*topSpherical,
		topLR_x,
		topLR_y,
		topImage,
		camera,
		0.333333f * M_PI,
		-1.666667f * M_PI,
		M_PI / 2.0f,
		M_PI / 2.0f - camera.getFov());

	// alpha feather the top spherical image for flow purposes
	cvtColor(*topSpherical, *topSpherical, CV_BGR2BGRA);
	const int yFeatherStart = topSpherical->rows - 1 - FLAGS_std_alpha_feather_size;
	for (int y = yFeatherStart; y < topSpherical->rows; ++y) {
		for (int x = 0; x < topSpherical->cols; ++x) {
			const float alpha =
				1.0f - float(y - yFeatherStart) / float(FLAGS_std_alpha_feather_size);
			topSpherical->at<Vec4b>(y, x)[3] = 255.0f * alpha;
		}
	}
}

float rampf(const float x, const float a, const float b) {
	if (min(1.0f, (x - a) / (b - a)) > 0)
		return min(1.0f, (x - a) / (b - a));
	else
		return 0;
}

void poleToSideFlowThread(
	string eyeName,
	const RigDescription& rig,
	Mat* sideSphericalForEye,
	Mat* fisheyeSpherical) {

	// crop the side panorama to the height of the pole image
	Mat croppedSideSpherical = (*sideSphericalForEye)(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
	//	croppedSideSpherical = featherAlphaChannel(croppedSideSpherical, FLAGS_std_alpha_feather_size);

	// extend the panoramas and wrap horizontally so we can avoid a seam
	const float kExtendFrac = 1.2f;
	const int extendedWidth = float(fisheyeSpherical->cols) * kExtendFrac;
	Mat extendedSideSpherical(Size(extendedWidth, fisheyeSpherical->rows), CV_8UC4);
	Mat extendedFisheyeSpherical(extendedSideSpherical.size(), CV_8UC4);
	for (int y = 0; y < extendedSideSpherical.rows; ++y) {
		for (int x = 0; x < extendedSideSpherical.cols; ++x) {
			extendedSideSpherical.at<Vec4b>(y, x) =
				croppedSideSpherical.at<Vec4b>(y, x % fisheyeSpherical->cols);
			extendedFisheyeSpherical.at<Vec4b>(y, x) =
				fisheyeSpherical->at<Vec4b>(y, x % fisheyeSpherical->cols);
		}
	}

	Mat flow;
	NovelViewUpdate flowAlg;
	flowAlg.computeOpticalFlow(
		extendedSideSpherical,
		extendedFisheyeSpherical,
		Mat(),
		Mat(),
		Mat(),
		flow);

	// make a ramp for alpha/flow magnitude
	const float kRampFrac = 1.0f; // fraction of available overlap used for ramp
	float poleCameraCropRadius;
	float poleCameraRadius;
	float sideCameraRadius;

	// use fov from bottom camera
	poleCameraRadius = rig.findCameraByDirection(-kGlobalUp).getFov();

	// use fov from first side camera
	sideCameraRadius = approximateFov(rig.rigSideOnly, true);

	// crop is average of side and pole cameras
	poleCameraCropRadius =
		0.5f * (M_PI / 2 - sideCameraRadius) +
		0.5f * (min(float(M_PI / 2), poleCameraRadius));

	// convert from radians to degrees
	poleCameraCropRadius *= 180 / M_PI;
	poleCameraRadius *= 180 / M_PI;
	sideCameraRadius *= 180 / M_PI;

	const float phiFromPole = poleCameraCropRadius;
	const float phiFromSide = 90.0f - sideCameraRadius;
	const float phiMid = (phiFromPole + phiFromSide) / 2.0f;
	const float phiDiff = fabsf(phiFromPole - phiFromSide);
	const float phiRampStart = phiMid - kRampFrac * phiDiff / 2.0f;
	const float phiRampEnd = phiMid + kRampFrac * phiDiff / 2.0f;

	cout << "phiMid" << phiMid << endl;
	cout << "phiRampEnd" << phiRampEnd << endl;


	Mat project_x, project_y;
	project_x.create(extendedFisheyeSpherical.size(), CV_32FC1);
	project_y.create(extendedFisheyeSpherical.size(), CV_32FC1);

	// ramp for flow magnitude
	//    1               for phi from 0 to phiRampStart
	//    linear drop-off for phi from phiRampStart to phiMid
	//    0               for phi from phiMid to totalRadius
	Mat warp(extendedFisheyeSpherical.size(), CV_32FC2);
	for (int y = 0; y < warp.rows; ++y) {
		const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
		const float alpha = 1.0f - rampf(phi, phiRampStart, phiMid);
		for (int x = 0; x < warp.cols; ++x) {
			warp.at<Point2f>(y, x) = Point2f(x, y) + (1.0f - alpha) * flow.at<Point2f>(y, x);

			project_x.at<float>(y, x) = x + (1.0f - alpha) * flow.at<Point2f>(y, x).x;
			project_y.at<float>(y, x) = y + (1.0f - alpha) * flow.at<Point2f>(y, x).y;
		}
	}

	//const string matDir = "Mat";
	//saveMatToFile(project_x, matDir + "/left_flow_x.xml");
	//saveMatToFile(project_y, matDir + "/left_flow_y.xml");
	topSide_x = project_x.clone();
	topSide_y = project_y.clone();

	Mat warpedExtendedFisheyeSpherical;
	remap(
		extendedFisheyeSpherical,
		warpedExtendedFisheyeSpherical,
		warp,
		Mat(),
		CV_INTER_CUBIC,
		BORDER_CONSTANT);

	Mat tmpramp = warpedExtendedFisheyeSpherical(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
	topRamp.create(tmpramp.size(), CV_32FC1);
	for (int y = 0; y < tmpramp.rows; ++y) {
		const float phi = poleCameraRadius * float(y + 0.5f) / float(warp.rows);
		const float alpha = 1.0f - rampf(phi, phiMid, phiRampEnd);
		for (int x = 0; x < tmpramp.cols; ++x) {
			topRamp.at<float>(y, x) = alpha;
		}
	}
}

void cudaPrepareTopImagesThread(
	const RigDescription& rig,
	const cuda::GpuMat pro_x,
	const cuda::GpuMat pro_y,
	const cuda::GpuMat top,
	cuda::GpuMat* topSpherical) {
	const Camera& camera = rig.findCameraByDirection(kGlobalUp);
	cuda::GpuMat tmp;
	tmp.create(
		FLAGS_eqr_height * camera.getFov() / M_PI,
		FLAGS_eqr_width,
		CV_8UC3);

	cuda::remap(top, tmp, pro_x, pro_y, CV_INTER_CUBIC, BORDER_CONSTANT);
	cuda::cvtColor(tmp, tmp, CV_BGR2BGRA);
	feather_alpha_chanel(tmp, *topSpherical, FLAGS_std_alpha_feather_size);
}

void cudaPoleToSideFlowThread(
	const RigDescription& rig,
	const cuda::GpuMat topramp,
	const cuda::GpuMat pro_x,
	const cuda::GpuMat pro_y,
	cuda::GpuMat* sideSphericalForEye,
	cuda::GpuMat* fisheyeSpherical,
	cuda::GpuMat* warpedSphericalForEye) {

	cuda::GpuMat croppedSideSpherical = (*sideSphericalForEye)(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));
	const float kExtendFrac = 1.2f;
	const int extendedWidth = float(fisheyeSpherical->cols) * kExtendFrac;
	cuda::GpuMat extendedSideSpherical(Size(extendedWidth, fisheyeSpherical->rows), CV_8UC4);
	cuda::GpuMat extendedFisheyeSpherical(extendedSideSpherical.size(), CV_8UC4);

	extendimage(
		croppedSideSpherical,
		extendedSideSpherical,
		*fisheyeSpherical,
		extendedFisheyeSpherical,
		extendedWidth);

	Mat tmpext;
	extendedFisheyeSpherical.download(tmpext);
	imwrite("image/tmpext.png", tmpext);

	cuda::GpuMat warpedExtendedFisheyeSpherical;
	cuda::remap(
		extendedFisheyeSpherical,
		warpedExtendedFisheyeSpherical,
		pro_x,
		pro_y,
		CV_INTER_CUBIC,
		BORDER_CONSTANT);

	*warpedSphericalForEye = warpedExtendedFisheyeSpherical(Rect(0, 0, fisheyeSpherical->cols, fisheyeSpherical->rows));

	Gpu_strip(*fisheyeSpherical, warpedExtendedFisheyeSpherical, *warpedSphericalForEye, *warpedSphericalForEye);

	alpha_cuda(topramp, *warpedSphericalForEye);

	cuda::copyMakeBorder(
		*warpedSphericalForEye,
		*warpedSphericalForEye,
		0,
		sideSphericalForEye->rows - warpedSphericalForEye->rows,
		0,
		0,
		BORDER_CONSTANT,
		Scalar(0, 0, 0, 0));
}

int main()
{
	const double readStart = getCurrTimeSec();
	WSAData wsaData;
	SOCKET listenfd, connfd;
	int seq_number;
	AVCodecContext *c = NULL;
	AVFrame *frame = NULL;
	AVPacket *pkt = NULL;
	uint8_t *pic_inbuff = NULL;
	FILE *fin, *fout;

	namedWindow("cam0", CV_WINDOW_NORMAL);
	
	while(1){
		camInit();
		RigDescription rig("camera_rig.json");
		firstCompute(rig);

		readInitMat();

		if (server_transfer_Init(&connfd, &listenfd, &wsaData) == -1) {
			printf("Socket error!\n");
			exit(1);
		}
		//注意传值和传址问题，初始化编码器
		x264_encoder_Init(&c, &frame, &pkt, &pic_inbuff, &seq_number);
		vector<cuda::GpuMat> projectionImages;
		const double readEnd = getCurrTimeSec();
		cout << "Read Init Mat Time: " << readEnd - readStart << endl;

		Mat tempsphericalImageL;
		cuda::GpuMat gtopSpherical, gcroppedSideSpherical, gpuextendedSideSpherical, gpuextendedFisheyeSpherical, gpuwarpedExtendedFisheyeSpherical, warpedSphericalForEye, gpudstL;

		Mat topImage;
		ready_for_topside();

		cuda::GpuMat gtopImage, gtopLR_x, gtopLR_y, gtopside_x, gtopside_y, gtopramp;


		bool stop = false;
		int count = 0;
		bool first = true;
		while (!stop) {
			const double starttime = getCurrTimeSec();
			projectSphericalCamImages(rig, "rgb", "000000", projectionImages);
			double opticalFlowRuntime, novelViewRuntime;
			cuda::GpuMat sphericalImageL, sphericalImageR;
			const double fovHorizontal =
				2 * approximateFov(rig.rigSideOnly, false) * (180 / M_PI);

			generateRingOfNovelViewsAndRenderStereoSpherical(
				rig.getRingRadius(),
				fovHorizontal,
				projectionImages,
				sphericalImageL,
				sphericalImageR,
				opticalFlowRuntime,
				novelViewRuntime);
			//Mat tmp;
			//sphericalImageL.download(tmp);
			//imwrite("tmpL.png", tmp);
			padToheight(sphericalImageL, FLAGS_eqr_height);

			CAM[6].read(topImage);
			//		imwrite("top.png", topImage);
			gtopImage.upload(topImage);

			if (first) {
				Mat firstImage;
				Mat topSpherical;
				sphericalImageL.download(firstImage);
				std::thread prepareTopThread, topFlowThreadL;
				prepareTopThread = std::thread(
					prepareTopImagesThread,
					cref(rig),
					topImage,
					&topSpherical);

				prepareTopThread.join();

				topFlowThreadL = std::thread(
					poleToSideFlowThread,
					"top_left",
					cref(rig),
					&firstImage,
					&topSpherical);

				topFlowThreadL.join();
				ready_for_cuda(&gtopImage, &gtopLR_x, &gtopLR_y, &gtopside_x, &gtopside_y, &gtopramp);
				first = false;
			}
			const double starttop = getCurrTimeSec();
			cuda::GpuMat cudaTop, topSphericalWarped;
			std::thread cudaPrepareTopThread, cudaTopFlowThread;
			cudaPrepareTopThread = std::thread(
				cudaPrepareTopImagesThread,
				cref(rig),
				gtopLR_x,
				gtopLR_y,
				gtopImage,
				&cudaTop);
			cudaPrepareTopThread.join();
			const double endtop = getCurrTimeSec();
			cout << "top prepare:" << endtop - starttop << endl;
			cudaTopFlowThread = std::thread(
				cudaPoleToSideFlowThread,
				cref(rig),
				gtopramp,
				gtopside_x,
				gtopside_y,
				&sphericalImageL,
				&cudaTop,
				&topSphericalWarped);

			cudaTopFlowThread.join();

			Gpu_flatten(sphericalImageL, topSphericalWarped, gpudstL);
			const double endtop2 = getCurrTimeSec();
			cout << "top stitch: " << endtop2 - endtop << endl;
			cuda::resize(
				gpudstL,
				gpudstL,
				Size(FLAGS_final_width, FLAGS_eqr_height),
				0,
				0,
				INTER_CUBIC);

			Mat result;
			gpudstL.download(result);
			cout << "download: " << getCurrTimeSec() - endtop2 << endl;
			imshow("cam0", result);

			Mat yuvImg;
			int bufLen = PIC_SIZE * 3 / 2;
			cvtColor(result, yuvImg, CV_BGRA2YUV_I420);
			memcpy(pic_inbuff, yuvImg.data, bufLen * sizeof(unsigned char));
			x264_encodeVideo(connfd, c, frame, pkt, pic_inbuff, count, fout);
			if (!flag_send)
				break;
			const double endtime = getCurrTimeSec();
			cout << "fps: " << 1.0f / (endtime - starttime) << endl;
			if (waitKey(1) == 27) { //按ESC键
				Mat top;
				stop = true;

				cout << "程序结束！" << endl;
				cout << "*** ***" << endl;
			}
			count++;
		}
		x264_encoder_Flush(connfd, c, pkt, fout);
		x264_encoder_Destroy(&c, &frame, &pkt, &pic_inbuff, fout);
		server_transfer_Destroy(&connfd, &listenfd);
	}
	

	return 0;
}