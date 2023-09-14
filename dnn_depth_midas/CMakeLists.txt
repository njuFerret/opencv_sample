# add_executable(dnn_depth_midas main.cpp depth_engine.cpp ) target_link_libraries(dnn_depth_midas common)

set(EXE_NAME dnn_depth_midas)

add_executable(${EXE_NAME} "")

target_sources(
  ${EXE_NAME}
  PRIVATE main.cpp depth_engine.cpp
  PUBLIC depth_engine.h # poor PUBLIC example, see discussion below for why
)

# 以下指定target : ${EXE_NAME} 的包含文件和链接库

# target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include" )

target_link_libraries(${EXE_NAME} PRIVATE common)
