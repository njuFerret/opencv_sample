set (EXE_NAME curve_fitting)

add_executable(${EXE_NAME} main.cpp)


## 以下指定target : ${EXE_NAME} 的包含文件和链接库

#target_include_directories(${EXE_NAME}
#PUBLIC
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include"
#)

target_link_libraries(${EXE_NAME} PRIVATE common)

