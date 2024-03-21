#include <experimental/filesystem>
#include <iostream>
#include <string>

#include "FaceDetector.h"
#include "PoseEstimator.h"
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    std::string rootPath = fs::path(__FILE__).parent_path().string();

    cv::Mat frame;
    if (argc < 2) {
        std::cout << "No image provided. Using the demo query image." << std::endl;
        frame = cv::imread(rootPath + "/example/query1.jpg");
    } else {
        frame = cv::imread(argv[1]);
    }

    // detect face
    FaceDetector faceDetector(rootPath + "/data/shape_predictor_68_face_landmarks.dat");
    cv::Rect bbox;
    std::vector<cv::Point> landmarks;
    faceDetector.detectFace(frame, bbox, landmarks);
    if (bbox.width == 0 || bbox.height == 0 || landmarks.empty()) {
        std::cout << "No faces detected" << std::endl;
        return 1;
    }

    // draw face
    faceDetector.drawFace(frame, bbox, landmarks);

    // estimate pose
    PoseEstimator poseEstimator(rootPath + "/data/generic_3D_68_face_landmarks.dat", frame.size());
    cv::Mat rvec, tvec;
    poseEstimator.solvePnP(landmarks, rvec, tvec);

    // draw pose
    poseEstimator.showPose(frame, rvec, tvec);
    return 0;
}
