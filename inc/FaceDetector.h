#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

class FaceDetector {
 public:
     explicit FaceDetector(const std::string &dlibPath);
     void detectFace(const cv::Mat &frame, cv::Rect &bbox, std::vector<cv::Point> &landmarks);
     void drawFace(const cv::Mat &frame, const cv::Rect &bbox, const std::vector<cv::Point> &landmarks);

 private:
     dlib::array2d<dlib::rgb_pixel> preprocessImage(const cv::Mat &image);
     dlib::frontal_face_detector _faceDetector;
     dlib::shape_predictor _landmarksDetector;
};
