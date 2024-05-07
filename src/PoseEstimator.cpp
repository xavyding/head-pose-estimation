#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "PoseEstimator.h"

static constexpr int SHOW_POSE_REAR_SIZE = 80;
static constexpr int SHOW_POSE_REAR_DEPTH = 0;
static constexpr int SHOW_POSE_FRONT_SIZE = 100;
static constexpr int SHOW_POSE_FRONT_DEPTH = 100;

PoseEstimator::PoseEstimator(const std::string &refLandmarksPath, const cv::Size &imageSize) : _imSize(imageSize) {
    // Load reference landmarks
    std::ifstream file(refLandmarksPath);
    for (int i = 0; i < 68; i++) {
        cv::Point3d point;
        file.read(reinterpret_cast<char*>(&point), sizeof(point));
        _refLandmarks3D.push_back(point);
    }
    file.close();
    int focalLength = _imSize.width;
    _cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength, 0.0, _imSize.width / 2.0,
                                              0.0, focalLength, _imSize.height / 2.0,
                                              0.0, 0.0, 1.0);
    _distCoeffs = cv::Mat(4, 1, cv::DataType<double>::type, cv::Scalar(0));
}

void PoseEstimator::solvePnP(const std::vector<cv::Point> &landmarkPoints, cv::Mat &rvec, cv::Mat &tvec) {
    std::vector<cv::Point2d> landmarks(landmarkPoints.begin(), landmarkPoints.end());
    cv::solvePnP(_refLandmarks3D, landmarks, _cameraMatrix, _distCoeffs, rvec, tvec);
}

cv::Vec3d PoseEstimator::computeEulerAngles(const cv::Mat &rvec, const cv::Mat &tvec) {
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    cv::Mat projectionMat;
    cv::hconcat(rmat, tvec, projectionMat);
    cv::Mat c, r, t, euler;
    cv::decomposeProjectionMatrix(projectionMat, c, r, t, cv::noArray(), cv::noArray(), cv::noArray(), euler);
    double pitch = euler.at<double>(0);
    double yaw = euler.at<double>(1);
    double roll = euler.at<double>(2);
    return cv::Vec3d(pitch, yaw, roll);  // in degree
}

void PoseEstimator::showPose(const cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec) {
    cv::Mat image = frame.clone();
    std::vector<cv::Point3d> points3d;
    points3d.push_back(cv::Point3d(-SHOW_POSE_REAR_SIZE, -SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_DEPTH));
    points3d.push_back(cv::Point3d(-SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_DEPTH));
    points3d.push_back(cv::Point3d(SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_DEPTH));
    points3d.push_back(cv::Point3d(SHOW_POSE_REAR_SIZE, -SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_DEPTH));
    points3d.push_back(cv::Point3d(-SHOW_POSE_REAR_SIZE, -SHOW_POSE_REAR_SIZE, SHOW_POSE_REAR_DEPTH));

    points3d.push_back(cv::Point3d(-SHOW_POSE_FRONT_SIZE, -SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_DEPTH));
    points3d.push_back(cv::Point3d(-SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_DEPTH));
    points3d.push_back(cv::Point3d(SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_DEPTH));
    points3d.push_back(cv::Point3d(SHOW_POSE_FRONT_SIZE, -SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_DEPTH));
    points3d.push_back(cv::Point3d(-SHOW_POSE_FRONT_SIZE, -SHOW_POSE_FRONT_SIZE, SHOW_POSE_FRONT_DEPTH));

    std::vector<cv::Point2d> points2d;
    cv::projectPoints(points3d, rvec, tvec, _cameraMatrix, _distCoeffs, points2d);

    std::vector<cv::Point> points;
    for (const auto &point : points2d) {
        points.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
    }
    cv::polylines(image, {points}, true, cv::Scalar(0, 255, 0), 2);
    cv::line(image, points[1], points[6], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
    cv::line(image, points[2], points[7], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
    cv::line(image, points[3], points[8], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
    cv::imshow("Pose Estimation", image);
    cv::waitKey(0);
}
