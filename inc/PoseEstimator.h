#pragma once
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

class PoseEstimator {
 public:
    explicit PoseEstimator(const std::string &refLandmarksPath, const cv::Size &imageSize);
    void solvePnP(const std::vector<cv::Point> &landmarks, cv::Mat &rvec, cv::Mat &tvec);
    void showPose(const cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec);
 private:
    std::vector<cv::Point3d> _refLandmarks3D;
    cv::Mat _cameraMatrix;
    cv::Size _imSize;
    cv::Mat _distCoeffs;  // not used (supposing no distortion)
};
