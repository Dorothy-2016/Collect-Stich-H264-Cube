#pragma once
#include <opencv2/opencv.hpp>
#include<iostream>  
using namespace std;
//using namespace cv;

struct LazyNovelViewBuffer {
	int width, height;
	// warpL[u][v] = (x, y, t). in the final panorama image at pixel coord u, v
	// we will take a piece of the novel view image at x, y, and time shift t.
	vector<vector<cv::Point3f>> warpL;
	vector<vector<cv::Point3f>> warpR;

	LazyNovelViewBuffer(int width, int height) {
		this->width = width;
		this->height = height;
		warpL = vector<vector<cv::Point3f>>(width, vector<cv::Point3f>(height));
		warpR = vector<vector<cv::Point3f>>(width, vector<cv::Point3f>(height));
	}
};

class NovelViewUpdate{
public:
	cv::Mat imageL, imageR;
	cv::Mat flowLtoR, flowRtoL;


	void computeOpticalFlow(
		const cv::Mat& rgba0byte,
		const cv::Mat& rgba1byte,
		const cv::Mat& prevFlow,
		const cv::Mat& prevI0BGRA,
		const cv::Mat& prevI1BGRA,
		cv::Mat& flow);

	void renderLazyNovelView(
		const int width,
		const int height,
		const vector<vector<cv::Point3f>>& novelViewWarpBuffer,
		const cv::Mat& srcImage,
		const cv::Mat& opticalFlow,
		const bool invertT,
		cv::Mat* warpComposition_x,
		cv::Mat* warpComposition_y, 
		cv::Mat* flowMag,
		cv::Mat* shiftT);

	void NovelViewUpdate::combineLazyNovelViews(
		const LazyNovelViewBuffer& lazyBuffer,
		cv::Mat* warpComposition_lx,
		cv::Mat* warpComposition_ly,
		cv::Mat* warpComposition_rx,
		cv::Mat* warpComposition_ry,
		cv::Mat* flowMag_l,
		cv::Mat* flowMag_r,
		cv::Mat* shift_l,
		cv::Mat* shift_r);


	void prepare(
		const cv::Mat& colorImageL,
		const cv::Mat& colorImageR,
		cv::Mat* prevFlowLtoR,
		cv::Mat* prevFlowRtoL,
		cv::Mat* prevColorImageL,
		cv::Mat* prevColorImageR);

	cv::Mat getFlowLtoR() { return flowLtoR; }
	cv::Mat getFlowRtoL() { return flowRtoL; }

	void setFlowLtoR(cv::Mat flow) { flowLtoR = flow; }
	void setFlowRtoL(cv::Mat flow) { flowRtoL = flow; }
};
