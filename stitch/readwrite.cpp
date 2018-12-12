#include "readwrite.h"

using namespace std;
using namespace cv;

void saveMatToFile(const Mat& flow, const string& filename) {
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "flow" << flow;
	fs.release();
}

Mat readMatFromFile(const string& filename) {
	FileStorage fs(filename, FileStorage::READ);
	Mat flow;
	fs["flow"] >> flow;
	fs.release();
	return flow;
}

void saveFlowToFile(const Mat& flow, const string& filename) {
	assert(flow.type() == CV_32FC2);
	int rows = flow.rows;
	int cols = flow.cols;
	FILE* file = fopen(filename.c_str(), "wb");
	if (file == NULL) {
		cout << "no file" << endl;
	}
	fwrite((void*)(&rows), sizeof(rows), 1, file);
	fwrite((void*)(&cols), sizeof(cols), 1, file);
	for (int y = 0; y < flow.rows; ++y) {
		for (int x = 0; x < flow.cols; ++x) {
			float fx = flow.at<Point2f>(y, x).x;
			float fy = flow.at<Point2f>(y, x).y;
			fwrite((void*)(&fx), sizeof(fx), 1, file);
			fwrite((void*)(&fy), sizeof(fy), 1, file);
		}
	}
	fclose(file);
}

Mat readFlowFromFile(const string& filename) {
	FILE* file = fopen(filename.c_str(), "rb");
	if (file == NULL) {
		cout << "no file" << endl;
	}
	int rows, cols;
	fread((void*)&rows, sizeof(rows), 1, file);
	fread((void*)&cols, sizeof(cols), 1, file);
	Mat flow(Size(cols, rows), CV_32FC2);
	for (int y = 0; y < flow.rows; ++y) {
		for (int x = 0; x < flow.cols; ++x) {
			float fx, fy;
			fread((void*)(&fx), sizeof(fx), 1, file);
			fread((void*)(&fy), sizeof(fy), 1, file);
			flow.at<Point2f>(y, x) = Point2f(fx, fy);
		}
	}
	fclose(file);
	return flow;
}