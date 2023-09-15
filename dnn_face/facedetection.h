#ifndef FACEDETECTION_H
#define FACEDETECTION_H
#include <opencv2/opencv.hpp>

// type of landmark in single face
using FaceLandmark = std::array<cv::Point, 5>;

//  type of single face info
typedef struct {
  cv::Rect facebbox;        // 脸部矩形
  FaceLandmark landmark;        // 脸部5个特征点，右眼->左眼坐标->鼻子坐标->右嘴角坐标->左嘴角坐标
  double confidence;        // 检测置信度
} face_t;

// type of all faces
using faces_t = std::vector<face_t>;

class FaceDetection {
public:
  FaceDetection(const std::string &model_path, const cv::Size &input_size = cv::Size(320, 320),
                float conf_threshold = 0.6f, float nms_threshold = 0.3f, int top_k = 5000, int backend_id = 0,
                int target_id = 0);

  void setBackendAndTarget(int backend_id, int target_id);

  /* Overwrite the input size when creating the model. Size format: [Width, Height].
   */
  void setInputSize(const cv::Size &input_size);

  /**
   * @brief getDetectedFacesMat 获取检测的人脸
   * @return 以cv::Mat形式保存的人脸数据
   */
  const cv::Mat &getDetectedFacesMat() const;
  /**
   * @brief getDetectedFaces 获取检测的人脸
   * @return 返回类型faces_t，以结构体数组形式存储的人类数据，每个元素为一个人脸数据结构体 fact_t
   */
  const faces_t &getDetectedFaces() const;

  /**
   * @brief infer 在输入图片中检测人脸
   * @param image => cv:Mat
   * @return  人脸检测结果，存储为 cv::Mat[num_faces,15]格式
   * faces	detection results stored in a 2D cv::Mat of shape [num_faces, 15]
   *     0-1: x, y of bbox top left corner                                          // 脸部矩形框的左上角坐标
   *     2-3: width, height of bbox                                                 // 脸部矩形框的宽度和高度
   *     4-5: x, y of right eye (blue point in the example image)                   // 右眼坐标
   *     6-7: x, y of left eye (red point in the example image)                     // 左眼坐标
   *     8-9: x, y of nose tip (green point in the example image)                   // 鼻子坐标
   *     10-11: x, y of right corner of mouth (pink point in the example image)     // 右嘴角坐标
   *     12-13: x, y of left corner of mouth (yellow point in the example image)    // 左嘴角坐标
   *     14: face score                                                             // 检测置信度
   */
  void infer(const cv::Mat &image);

  void populateFaces(const cv::Mat &faces_mat);

private:
  cv::Ptr<cv::FaceDetectorYN> model;

  std::string model_path_;
  cv::Size input_size_;
  float conf_threshold_;
  float nms_threshold_;
  int top_k_;
  int backend_id_;
  int target_id_;
  faces_t faces_detected;
  cv::Mat faces_mat_detected;
};

// cv::Mat visualize(const cv::Mat &src_image, const cv::Mat &faces, float fps = -1.f);
cv::Mat visualize(const cv::Mat &src_image, const faces_t &faces, float fps = -1.f);

#endif        // FACTDETECTION_H
