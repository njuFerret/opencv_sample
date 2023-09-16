set (EXE_NAME transformation_homography)

add_executable(${EXE_NAME} "")

## 以下指定target : ${EXE_NAME} 的包含文件和链接库
target_sources(
  ${EXE_NAME}
  PRIVATE main.cpp
  #PUBLIC depth_engine.h # poor PUBLIC example, see discussion below for why
)

#target_include_directories(${EXE_NAME}
#PUBLIC
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include"
#)

target_link_libraries(${EXE_NAME} PRIVATE common)
