#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "FaceDetector.h"

#include "opencv2/core/core_c.h"
#include "opencv2/videoio/legacy/constants_c.h"
#include "opencv2/highgui/highgui_c.h"

FaceDetector::FaceDetector(const std::string &shapePredictorPath) {
    _faceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(shapePredictorPath) >> _landmarksDetector;
}

dlib::array2d<dlib::rgb_pixel> FaceDetector::preprocessImage(const cv::Mat &frame) {
    cv::Mat image = frame;
    if (image.empty()) {
        std::cerr << "Error: Image is empty." << std::endl;
        exit(1);
    }
    if (image.type() != CV_8UC3) cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    IplImage ipl_img = cvIplImage(image);
    dlib::cv_image<dlib::rgb_pixel> image2(&ipl_img);
    dlib::array2d<dlib::rgb_pixel> dlibImage;
    assign_image(dlibImage, image2);
    return dlibImage;
}

void FaceDetector::detectFace(const cv::Mat &frame, cv::Rect &bbox, std::vector<cv::Point> &landmarks) {
    dlib::array2d<dlib::rgb_pixel> dlibImage = preprocessImage(frame);
    std::vector<dlib::rectangle> dets = _faceDetector(dlibImage);
    if (dets.empty()) return;
    dlib::rectangle dlibBB = dets[0];
    bbox = cv::Rect(dlibBB.left(), dlibBB.top(), dlibBB.width(), dlibBB.height());
    dlib::full_object_detection faceShape = _landmarksDetector(dlibImage, dlibBB);
    landmarks.clear();
    for (int i = 0; i < faceShape.num_parts(); i++) {
        dlib::point ptdlib = faceShape.part(i);
        landmarks.push_back(cv::Point(ptdlib.x(), ptdlib.y()));
    }
}

void FaceDetector::drawFace(const cv::Mat &frame, const cv::Rect &bbox, const std::vector<cv::Point> &landmarks) {
    cv::Mat image = frame.clone();
    cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 2);
    for (const auto &point : landmarks) {
        cv::circle(image, point, 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::imshow("Face Detection", image);
}
