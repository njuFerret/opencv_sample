#include <opencv2/opencv.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"
#include <camera_model.h>
#include <string>

// #define WINDOW1_NAME "Window 1"
constexpr std::string kWindowMain = "WindowMain";
static constexpr char kWindowParam[] = "WindowParam";
static constexpr int32_t kWidth = 1280;
static constexpr int32_t kHeight = 720;
static constexpr float kFovDeg = 80.0f;

static CameraModel camera;

/*** Function ***/
void ResetCameraPose() {
  camera.SetExtrinsic(
      {0.0f, 0.0f, 0.0f}, /* rvec [deg] */
      //{ 0.0f, 0.0f, 7.0f }, false);   /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */
      {0.0f, 0.0f, -7.0f}, true); /* tvec (Oc - Ow in world coordinate. X+= Right, Y+ = down, Z+ = far) */
}

void ResetCamera(int32_t width, int32_t height) {
  camera.SetIntrinsic(width, height, FocalLength(width, kFovDeg));
  camera.SetDist({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  ResetCameraPose();
}

void CallbackMouseMain(int32_t event, int32_t x, int32_t y, int32_t flags, void *userdata) {
  static constexpr float kIncAnglePerPx = 0.1f;
  static constexpr int32_t kInvalidValue = -99999;
  static cv::Point s_drag_previous_point = {kInvalidValue, kInvalidValue};
  if (event == cv::EVENT_LBUTTONUP) {
    s_drag_previous_point.x = kInvalidValue;
    s_drag_previous_point.y = kInvalidValue;
  } else if (event == cv::EVENT_LBUTTONDOWN) {
    s_drag_previous_point.x = x;
    s_drag_previous_point.y = y;
  } else {
    if (s_drag_previous_point.x != kInvalidValue) {
      float delta_yaw = kIncAnglePerPx * (x - s_drag_previous_point.x);
      float pitch_delta = -kIncAnglePerPx * (y - s_drag_previous_point.y);
      camera.RotateCameraAngle(pitch_delta, delta_yaw, 0);
      s_drag_previous_point.x = x;
      s_drag_previous_point.y = y;
    }
  }
}

#define MAKE_GUI_SETTING_FLOAT(VAL, LABEL, STEP, FORMAT, RANGE0, RANGE1)                                               \
  {                                                                                                                    \
    cvui::beginColumn(-1, -1, 2);                                                                                      \
    double temp_double_current = static_cast<double>(VAL);                                                             \
    double temp_double_new = temp_double_current;                                                                      \
    float temp_float_current = VAL;                                                                                    \
    float temp_float_new = temp_float_current;                                                                         \
    cvui::text(LABEL);                                                                                                 \
    cvui::counter(&temp_double_new, STEP, FORMAT);                                                                     \
    cvui::trackbar<float>(200, &temp_float_new, RANGE0, RANGE1);                                                       \
    if (temp_double_new != temp_double_current)                                                                        \
      VAL = static_cast<float>(temp_double_new);                                                                       \
    if (temp_float_new != temp_float_current)                                                                          \
      VAL = temp_float_new;                                                                                            \
    cvui::endColumn();                                                                                                 \
  }

//#define MAKE_GUI_SETTING_FLOAT(VAL, LABEL, STEP, FORMAT, RANGE0, RANGE1)                                               \
//  {                                                                                                                    \
//    cvui::beginColumn(-1, -1, 2);                                                                                      \
//    double temp_double_current = static_cast<double>(VAL);                                                             \
//    double temp_double_new = temp_double_current;                                                                      \
//    float temp_float_current = VAL;                                                                                    \
//    float temp_float_new = temp_float_current;                                                                         \
//    cvui::text(LABEL);                                                                                                 \
//    cvui::counter(&temp_double_new, STEP, FORMAT);                                                                     \
//    cvui::trackbar<float>(200, &temp_float_new, RANGE0, RANGE1);                                                       \
//    if (temp_double_new != temp_double_current)                                                                        \
//      VAL = static_cast<float>(temp_double_new);                                                                       \
//    if (temp_float_new != temp_float_current)                                                                          \
//      VAL = temp_float_new;                                                                                            \
//    cvui::endColumn();                                                                                                 \
//  }

void loop_main(const cv::Mat &image_org) {
  cvui::context(kWindowMain);

  /* Generate object points (3D: world coordinate) */
  std::vector<cv::Point3f> object_point_list;
  float aspect = static_cast<float>(image_org.cols) / image_org.rows;
  object_point_list.push_back(cv::Point3f(-1 * aspect, -1, 0));
  object_point_list.push_back(cv::Point3f(1 * aspect, -1, 0));
  object_point_list.push_back(cv::Point3f(1 * aspect, 1, 0));
  object_point_list.push_back(cv::Point3f(-1 * aspect, 1, 0));

  /* Rotate object */
  static float x_deg, y_deg, r_deg;
  CameraModel::RotateObject(x_deg, y_deg, r_deg, object_point_list);
  x_deg += 1.1f; /* don't use a good cut-off number to avoid gimbal lock */
  y_deg += 1.2f;
  r_deg += 1.3f;

  /* Convert to image points (2D) */
  std::vector<cv::Point2f> image_point_list;
  cv::projectPoints(object_point_list, camera.rvec, camera.tvec, camera.K, camera.dist_coeff, image_point_list);

  /* Affine transform */
  cv::Point2f pts1[] = {cv::Point2f(0, 0), cv::Point2f(image_org.cols - 1.0f, 0),
                        cv::Point2f(image_org.cols - 1.0f, image_org.rows - 1.0f),
                        cv::Point2f(0, image_org.rows - 1.0f)};
  cv::Mat mat_affine = cv::getPerspectiveTransform(pts1, &image_point_list[0]);
  cv::Mat mat_output = cv::Mat(kHeight, kWidth, CV_8UC3, cv::Scalar(70, 70, 70));
  cv::warpPerspective(image_org, mat_output, mat_affine, mat_output.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

  cvui::imshow(kWindowMain, mat_output);
}

void loop_param() {
  cvui::context(kWindowParam);
  cv::Mat mat = cv::Mat(1000, 300, CV_8UC3, cv::Scalar(70, 70, 70));

  cvui::beginColumn(mat, 10, 10, -1, -1, 2);
  {
    cvui::text("Reset Camera Parameter");
    if (cvui::button(120, 20, "Reset")) {
      ResetCameraPose();
    }

    cvui::text("Camera Parameter (Intrinsic)");

    MAKE_GUI_SETTING_FLOAT(camera.fx(), "Focal Length", 10.0f, "%.0f", 0.0f, 1000.0f);
    camera.fy() = camera.fx();

    MAKE_GUI_SETTING_FLOAT(camera.dist_coeff.at<float>(0), "dist: k1", 0.00001f, "%.05f", -0.4f, 0.4f);
    MAKE_GUI_SETTING_FLOAT(camera.dist_coeff.at<float>(1), "dist: k2", 0.00001f, "%.05f", -0.1f, 0.1f);
    MAKE_GUI_SETTING_FLOAT(camera.dist_coeff.at<float>(2), "dist: p1", 0.00001f, "%.05f", -0.1f, 0.1f);
    MAKE_GUI_SETTING_FLOAT(camera.dist_coeff.at<float>(3), "dist: p2", 0.00001f, "%.05f", -0.1f, 0.1f);
    MAKE_GUI_SETTING_FLOAT(camera.dist_coeff.at<float>(4), "dist: k3", 0.00001f, "%.05f", -0.1f, 0.1f);

    camera.UpdateNewCameraMatrix();

    cvui::text("Camera Parameter (Extrinsic)");
    float pitch_deg = Rad2Deg(camera.rx());
    MAKE_GUI_SETTING_FLOAT(pitch_deg, "Pitch", 1.0f, "%.0f", -90.0f, 90.0f);
    float yaw_deg = Rad2Deg(camera.ry());
    MAKE_GUI_SETTING_FLOAT(yaw_deg, "Yaw", 1.0f, "%.0f", -90.0f, 90.0f);
    float roll_deg = Rad2Deg(camera.rz());
    MAKE_GUI_SETTING_FLOAT(roll_deg, "Roll", 1.0f, "%.0f", -90.0f, 90.0f);
    camera.SetCameraAngle(pitch_deg, yaw_deg, roll_deg);

    float x = -camera.tx();
    float y = -camera.ty();
    float z = -camera.tz();
    MAKE_GUI_SETTING_FLOAT(x, "X", 1.0f, "%.0f", -20.0f, 20.0f);
    MAKE_GUI_SETTING_FLOAT(y, "Y", 1.0f, "%.0f", -20.0f, 20.0f);
    MAKE_GUI_SETTING_FLOAT(z, "Z", 1.0f, "%.0f", -20.0f, 20.0f);
    camera.SetCameraPos(x, y, z, false);
  }
  cvui::endColumn();

  cvui::imshow(kWindowParam, mat);
}

int main() {
  cvui::init(kWindowMain);
  cvui::init(kWindowParam);
  cv::setMouseCallback(kWindowMain, CallbackMouseMain);

  static const std::string image_path = RESOURCE_DIR "baboon.jpg";
  cv::Mat image_org = cv::imread(image_path);

  ResetCamera(kWidth, kHeight);
  cv::Mat frame = image_org;

  //  while (true) {
  //    //    frame = cv::Scalar(49, 52, 49);
  //    cvui::text(frame, 50, 50, "Hello world!");

  //    cvui::imshow(kWindowMain, frame);

  //    if (cv::waitKey(20) == 27) {
  //      break;
  //    }
  //  }

  while (true) {
    loop_main(image_org);
    loop_param();
    int32_t key = cv::waitKey(1);
    if (key == 27)
      break; /* ESC to quit */
             //      TreatKeyInputMain(key);
  }
  return 0;
}
