#include "RigDescription.h"
#include <fstream>

namespace surround360 {

	using namespace std;
	using namespace cv;
	RigDescription::RigDescription(const string& filename) {
		rig = Camera::loadRig(filename);
		for (const Camera& camera : rig) {
			if (camera.group.find("side") != string::npos) {
				rigSideOnly.emplace_back(camera);
			}
		}

		// validation
//		CHECK_NE(getSideCameraCount(), 0);
	}

	const Camera& RigDescription::findCameraByDirection(
		const Camera::Vector3& direction,
		const Camera::Real distCamAxisToRigCenterMax) const {
		const Camera* best = nullptr;
		for (const Camera& camera : rig) {
			if (best == nullptr ||
				best->forward().dot(direction) < camera.forward().dot(direction)) {
				if (distCamAxisToRigCenter(camera) <= distCamAxisToRigCenterMax) {
					best = &camera;
				}
			}
		}
//		return *CHECK_NOTNULL(best);
		return *best;
	}

	// find the camera with the largest distance from camera axis to rig center
	const Camera& RigDescription::findLargestDistCamAxisToRigCenter() const {
		const Camera* best = &rig.back();
		for (const Camera& camera : rig) {
			if (distCamAxisToRigCenter(camera) > distCamAxisToRigCenter(*best)) {
				best = &camera;
			}
		}
		return *best;
	}

	string RigDescription::getTopCameraId() const {
		return findCameraByDirection(Camera::Vector3::UnitZ()).id;
	}

	string RigDescription::getBottomCameraId() const {
		return findCameraByDirection(-Camera::Vector3::UnitZ()).id;
	}

	string RigDescription::getBottomCamera2Id() const {
		return findLargestDistCamAxisToRigCenter().id;
	}

	int RigDescription::getSideCameraCount() const {
		return rigSideOnly.size();
	}

	string RigDescription::getSideCameraId(const int idx) const {
		return rigSideOnly[idx].id;
	}

	float RigDescription::getRingRadius() const {
		return rigSideOnly[0].position.norm();
	}
	
	void imreadInStdThread(Mat frame, int flags, cuda::GpuMat* gpudst, cuda::Stream stream) {
		if (frame.empty()) {
			cout << "image empty" << endl;
			exit(-1);
		}
		gpudst->upload(frame, stream);
		stream.waitForCompletion();
	}

	vector<cuda::GpuMat> RigDescription::loadSideCameraImages(
		Mat* frame,
		cuda::Stream* stream) const {

//		cout << "loadSideCameraImages spawning threads" << endl;
		vector<std::thread> threads;
		vector<cuda::GpuMat> images(getSideCameraCount());
		for (int i = 0; i < getSideCameraCount(); ++i) {
			threads.emplace_back(
				imreadInStdThread,
				frame[i],
				CV_LOAD_IMAGE_COLOR,
				&(images[i]),
				stream[i]);
		}
		for (auto& thread : threads) {
			thread.join();
		}
		return images;
	}

} // namespace surround360