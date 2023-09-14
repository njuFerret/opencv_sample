set (EXE_NAME reconstruction_depth_to_3d)


#file(COPY "${CMAKE_CURRENT_LIST_DIR}/model-small.onnx" DESTINATION resource/model FOLLOW_SYMLINK_CHAIN)

function (get_file_name_and_path FULL_FILE_PATH OUT_PATH OUT_FILENAME)
    # 路径中的 \\ 替换为 /
    STRING(REPLACE "\\" "/" file_name_normaled "${FULL_FILE_PATH}")
    # 获得文件名(xxx/yyy/ccc.d => ccc.d)
    STRING(REGEX REPLACE ".+/(.+\\..*)" "\\1" NName "${file_name_normaled}")
    # 获得目标路径
    STRING(REPLACE "/${NName}" "" PPath "${file_name_normaled}")

    # 返回值，必须要有 PARENT_SCOPE
    set(${OUT_FILENAME} ${NName} PARENT_SCOPE)
    set(${OUT_PATH} ${PPath} PARENT_SCOPE)

endfunction(get_file_name_and_path)


#resource/model/midasv2_small_256x256.onnx
function (CopyAndRename source_file target_file)
    get_file_name_and_path(${source_file} src_file_dir src_file_name)
#    set (src_file_name ${OUT_FILENAME})
#    set (src_file_dir ${OUT_PATH})


    get_file_name_and_path(${target_file} target_file_dir target_file_name)
#    set (target_file_name ${OUT_FILENAME})
#    set (target_file_dir ${OUT_PATH})
    message(${src_file_dir} **/** ${src_file_name} :==: ${target_file_dir}*=/=* ${target_file_name})

    file(COPY "${source_file}" DESTINATION "${target_file_dir}" FOLLOW_SYMLINK_CHAIN)
#    file(RENAME "${target_file_dir}/${src_file_name}"  "${target_file_dir}/${target_file_name}")

endfunction(CopyAndRename)

CopyAndRename("${CMAKE_CURRENT_LIST_DIR}/model-small.onnx" "resource/model/midasv2_small_256x256.onnx")



add_executable(${EXE_NAME} main.cpp depth_engine.cpp depth_engine.h)


## 以下指定target : ${EXE_NAME} 的包含文件和链接库

#target_include_directories(${EXE_NAME}
#PUBLIC
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/freetype/include/freetype2"
# "${CMAKE_CURRENT_LIST_DIR}/../3rdParty/harfbuzz/include"
#)

target_link_libraries(${EXE_NAME} PRIVATE common)



