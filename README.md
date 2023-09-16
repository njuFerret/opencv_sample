
# update for OpenCV 4.8.0

迁移至OpenCV 4.8.0, YuNet(face_detection_yunet_2023mar.onnx)编译通过

> platform:
>     windows 10 x64
>     MinGW-w64-x86_64-13.1.0-release-posix-seh-ucrt-rt_v11-rev1
>     opencv 4.8.0
>     cmake 3.27.4
>     qtcreator 11.0.2 (qt 6.5.2)

编译指令：
```dos
cd <opencv_sample_dir>
cmake -GNinja -DOpenCV_DIR=D:/Dev/Qt/6.5.2/3rdParty/opencv -Bbuild -S.
cmake --build build --parallel

```

关于人脸识别示例

使用`opencv model zoo`中的`face_detection_yunet`重新改写原示例中的人脸识别部分，同时更新模型，测试使用`model zoo`中的`face_detection_yunet_2023mar.onnx`模型和`yunet_n_640_640.onnx`模型(南方科技大学于仕琪老师)均通过，[链接地址](https://github.com/ShiqiYu/libfacedetection.train/tree/master/onnx)

# OpenCV Sample Code
https://user-images.githubusercontent.com/11009876/131844621-a7c43048-edd7-4f74-b85d-7c7bf74f53bb.mp4


## undistortion_calibration
![00_doc/undistortion_calibration.jpg](00_doc/undistortion_calibration.jpg)

- Basic camera calibration using chessboard pattern


## undistortion_manual_unified_projection
![00_doc/undistortion_manual_unified_projection.jpg](00_doc/undistortion_manual_unified_projection.jpg)

- Manual camera calibration using the unified projection model for fisheye / omnidirectional camera

## projection_points_3d_to_2d
- Projection (3D points (world coordinate) to a 2D image plane) using editable camera parameters

https://user-images.githubusercontent.com/11009876/131841810-3901988e-223f-4ec8-967d-372c6a53de04.mp4

https://user-images.githubusercontent.com/11009876/131841829-4c38c713-fa71-4997-a9df-ca5e35ce348d.mp4


## projection_image_3d_to_2d
- Projection (image in world coordinate to a 2D image plane) using editable camera parameters

https://user-images.githubusercontent.com/11009876/131841863-bee26051-f8c0-4dd2-a31c-a4b8f1df722b.mp4

## transformation_topview_projection
- Transformation to top view image using projection

![00_doc/transformation_topview_projection.jpg](00_doc/transformation_topview_projection.jpg)

## transformation_homography
- Homobraphy transformation

https://user-images.githubusercontent.com/11009876/132087422-79d37de1-3ea4-476c-88c1-eb53cf88b492.mp4

## distance_calculation
- Distance calculation on ground plane

![00_doc/distance_calculation.jpg](00_doc/distance_calculation.jpg)

## dnn_face
- Face Detection using YuNet
- Head Pose Estimatino Using SolvePnP
- Overlay icon with transparent mask

![00_doc/dnn_face.jpg](00_doc/dnn_face.jpg)
![00_doc/dnn_face_mask.jpg](00_doc/dnn_face_mask.jpg)

## dnn_depth_midas
- Depth estimation using MiDaS small V2.1
- You need to download the model
    - from: https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx
    - to: `resource/mdoel/midasv2_small_256x256.onnx`

https://user-images.githubusercontent.com/11009876/144711379-a3d4b3c4-86e9-4b33-a90e-b4ac0eb584e2.mp4


## reconstruction_depth_to_3d
- 3D Reconstruction
    - Generate 3D point cloud from one single still image using depth map
    - Project these points onto 2D image with a virtual camera

https://user-images.githubusercontent.com/11009876/144705856-8714558e-610f-4087-a194-11e712517b9f.mp4

# License
- Copyright 2021 iwatake2222
- Licensed under the Apache License, Version 2.0
    - [LICENSE](LICENSE)

# Acknowledgements
- cvui
    - https://github.com/Dovyski/cvui
    - Copyright (c) 2016 Fernando Bevilacqua
    - Licensed under the MIT License (MIT)
    - Source code is copied
- OpenCV
    - https://github.com/opencv/opencv
    - Licensed under the Apache License, Version 2.0
    - Image files are copied
- OpenCV Zoo
    - https://github.com/opencv/opencv_zoo
    - Licensed under the Apache License, Version 2.0
    - Model files are copied
- MiDaS
    - https://github.com/isl-org/MiDaS
    - Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)
    - Licensed under the MIT License
- https://pixabay.com/ja/
    - room_00.jpg, room_01.jpg, room_02.jpg
- https://www.photo-ac.com
    - fisheye_00.jpg
- YouTube
    - dashcam_00.jpg (Copyright Dashcam Roadshow 2020. https://www.youtube.com/watch?v=tTuUjnISt9s )
- Others
    - https://lovelive-as.bushimo.jp/wp-content/uploads/2019/09/chara_09_stage.png (https://lovelive-as.bushimo.jp/member/rina/ )
