#ifndef FACEDETECTION_H
#define FACEDETECTION_H
#include <opencv2/opencv.hpp>

class FaceDetection {
public:
  FaceDetection(const std::string &model_path, const cv::Size &input_size = cv::Size(320, 320),
                float conf_threshold = 0.6f, float nms_threshold = 0.3f, int top_k = 5000, int backend_id = 0,
                int target_id = 0);

  void setBackendAndTarget(int backend_id, int target_id);

  /* Overwrite the input size when creating the model. Size format: [Width, Height].
   */
  void setInputSize(const cv::Size &input_size);

  cv::Mat infer(const cv::Mat &image);

private:
  cv::Ptr<cv::FaceDetectorYN> model;

  std::string model_path_;
  cv::Size input_size_;
  float conf_threshold_;
  float nms_threshold_;
  int top_k_;
  int backend_id_;
  int target_id_;
};

cv::Mat visualize(const cv::Mat &image, const cv::Mat &faces, float fps = -1.f);

#endif        // FACTDETECTION_H
