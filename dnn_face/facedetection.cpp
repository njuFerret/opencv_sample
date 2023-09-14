#include "facedetection.h"

const std::map<std::string, int> str2backend{{"opencv", cv::dnn::DNN_BACKEND_OPENCV},
                                             {"cuda", cv::dnn::DNN_BACKEND_CUDA},
                                             {"timvx", cv::dnn::DNN_BACKEND_TIMVX},
                                             {"cann", cv::dnn::DNN_BACKEND_CANN}};
const std::map<std::string, int> str2target{{"cpu", cv::dnn::DNN_TARGET_CPU},
                                            {"cuda", cv::dnn::DNN_TARGET_CUDA},
                                            {"npu", cv::dnn::DNN_TARGET_NPU},
                                            {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}};

FaceDetection::FaceDetection(const std::string &model_path, const cv::Size &input_size, float conf_threshold,
                             float nms_threshold, int top_k, int backend_id, int target_id)
    : model_path_(model_path), input_size_(input_size), conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
      top_k_(top_k), backend_id_(backend_id), target_id_(target_id) {
  model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_,
                                     target_id_);
}

void FaceDetection::setBackendAndTarget(int backend_id, int target_id) {
  backend_id_ = backend_id;
  target_id_ = target_id;
  model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_,
                                     target_id_);
}

void FaceDetection::setInputSize(const cv::Size &input_size) {
  input_size_ = input_size;
  model->setInputSize(input_size_);
}

cv::Mat FaceDetection::infer(const cv::Mat &image) {
  cv::Mat res;
  model->detect(image, res);
  return res;
}

cv::Mat visualize(const cv::Mat &image, const cv::Mat &faces, float fps) {
  static const cv::Scalar box_color{0, 255, 0};
  static std::vector<cv::Scalar> landmark_color{
      cv::Scalar(255, 0, 0),          // right eye
      cv::Scalar(0, 0, 255),          // left eye
      cv::Scalar(0, 255, 0),          // nose tip
      cv::Scalar(255, 0, 255),        // right mouth corner
      cv::Scalar(0, 255, 255)         // left mouth corner
  };
  static const cv::Scalar text_color{0, 255, 0};

  auto output_image = image.clone();

  if (fps >= 0) {
    cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color,
                2);
  }

  for (int i = 0; i < faces.rows; ++i) {
    // Draw bounding boxes
    int x1 = static_cast<int>(faces.at<float>(i, 0));
    int y1 = static_cast<int>(faces.at<float>(i, 1));
    int w = static_cast<int>(faces.at<float>(i, 2));
    int h = static_cast<int>(faces.at<float>(i, 3));
    cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

    // Confidence as text
    float conf = faces.at<float>(i, 14);
    cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5,
                text_color);

    // Draw landmarks
    for (int j = 0; j < landmark_color.size(); ++j) {
      float x = faces.at<float>(i, 2 * j + 4);
      float y = faces.at<float>(i, 2 * j + 5);
      //        int x = static_cast<int>(faces.at<float>(i, 2 * j + 4));
      //        int y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
      cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
    }
  }
  return output_image;
}
