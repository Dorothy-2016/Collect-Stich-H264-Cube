#pragma once

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <string>
#include <iostream>

using namespace std;


void saveMatToFile(const cv::Mat& flow, const string& filename);

cv::Mat readMatFromFile(const string& filename);

cv::Mat readFlowFromFile(const string& filename);

void saveFlowToFile(const cv::Mat& flow, const string& filename);