# add_executable(dnn_face  ) target_link_libraries(dnn_face common)

set(EXE_NAME dnn_face)

add_executable(${EXE_NAME} "")

target_sources(
  ${EXE_NAME}
  PRIVATE main.cpp facedetection.cpp # face_detection.cpp
  PUBLIC facedetection.h # face_detection.h # poor PUBLIC example, see discussion below for why
)

# 以下指定target : ${EXE_NAME} 的包含文件和链接库

# target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include" )

target_link_libraries(${EXE_NAME} PRIVATE common)
