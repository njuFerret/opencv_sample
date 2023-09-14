# 编译静态库，暂不添加任何源码。
set(_LIB_NAME common)

add_library(${_LIB_NAME} "")

# 随后用target_sources添加源码。
# 这种方式更好，因为可以更细粒度地访问权限。
# 例如对cpp文件就是PRIVATE，而hpp则是PUBLIC
target_sources(${_LIB_NAME}
PRIVATE
    # ${CMAKE_CURRENT_LIST_DIR}代表当前的CMakeLists.txt文件所在的绝对路径。
    ${CMAKE_CURRENT_LIST_DIR}/camera_model.cpp
    ${CMAKE_CURRENT_LIST_DIR}/common_helper_cv.cpp 
#    ${CMAKE_CURRENT_LIST_DIR}/CvxFont.cpp              # 显示中文，现在使用opencv_contrib.freetype 替代
PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/common_helper_cv.h
    ${CMAKE_CURRENT_LIST_DIR}/camera_model.h 
    ${CMAKE_CURRENT_LIST_DIR}/curve_fitting.h 
#    ${CMAKE_CURRENT_LIST_DIR}/CvxFont.h                # 显示中文，现在使用opencv_contrib.freetype 替代
)

### message("${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2")

#target_include_directories(${_LIB_NAME}                # 显示中文， 包含的头文件路径
#PUBLIC
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include"
#)


##target_link_libraries(${_LIB_NAME} )                   # 显示中文，链接库，这里没有链接库，在具体子程序里连接



