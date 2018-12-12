#pragma once
#include <opencv2/opencv.hpp>
#include<iostream>  
#include "readwrite.h"
using namespace std;


class NovelViewGenerator{
public:
	cv::cuda::GpuMat imageL, imageR;
	cv::Mat flowLtoR, flowRtoL;

	cv::cuda::GpuMat renderLazyNovelView(
		const cv::cuda::GpuMat& srcImage,
		const bool invertT,
		const cv::cuda::GpuMat& warpComposition_x,
		const cv::cuda::GpuMat& warpComposition_y,
		const cv::cuda::GpuMat& shift,
		cv::cuda::Stream stream);

	cv::cuda::GpuMat combineLazyNovelViews(
		const cv::cuda::GpuMat& warpComposition_lx,
		const cv::cuda::GpuMat& warpComposition_ly,
		const cv::cuda::GpuMat& warpComposition_rx,
		const cv::cuda::GpuMat& warpComposition_ry,
		const cv::cuda::GpuMat& flowMag_l,
		const cv::cuda::GpuMat& flowMat_r,
		const cv::cuda::GpuMat& shift_l,
		const cv::cuda::GpuMat& shift_r,
		cv::cuda::Stream streaml,
		cv::cuda::Stream streamr);


	void prepare(
		const cv::cuda::GpuMat& colorImageL,
		const cv::cuda::GpuMat& colorImageR);

	cv::Mat getFlowLtoR() { return flowLtoR; }
	cv::Mat getFlowRtoL() { return flowRtoL; }

	void setFlowLtoR(cv::Mat flow) { flowLtoR = flow; }
	void setFlowRtoL(cv::Mat flow) { flowRtoL = flow; }
};
