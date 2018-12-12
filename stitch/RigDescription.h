#pragma once
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "Camera.h"


namespace surround360 {
	using namespace std;
	struct RigDescription {
		Camera::Rig rig;
		Camera::Rig rigSideOnly;
		RigDescription(const string& filename);

		// find the camera that is closest to pointing in the provided direction
		// ignore those with excessive distance from the camera axis to the rig center
		const Camera& findCameraByDirection(
			const Camera::Vector3& direction,
			const Camera::Real distCamAxisToRigCenterMax = 1.0) const;

		// find the camera with the largest distance from camera axis to rig center
		const Camera& findLargestDistCamAxisToRigCenter() const;

		string getTopCameraId() const;

		string getBottomCameraId() const;

		string getBottomCamera2Id() const;

		int getSideCameraCount() const;

		string getSideCameraId(const int idx) const;

		float getRingRadius() const;

		vector<cv::cuda::GpuMat> loadSideCameraImages(
			cv::Mat* frame,
			cv::cuda::Stream* stream) const;

	private:
		static Camera::Real distCamAxisToRigCenter(const Camera& camera) {
			return camera.rig(camera.principal).distance(Camera::Vector3::Zero());
		}
	};

} // namespace surround360
