# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(object_tracker VERSION 0.0.1 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 寻找OpenCV库，使用自定义的OpenCV_DIR
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_VERSION}")

#set(cv_bridge_DIR "/opt/ros2/humble/local/lib/python3.10/dist-packages/cv_bridge")
# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
#find_package(rmw REQUIRED)
#find_package(rcl_interfaces REQUIRED) 
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(geometry_msgs REQUIRED)



# 寻找Eigen3
set(EIGEN_VERSION 3.3)
find_package(Eigen3 ${EIGEN_VERSION} QUIET)


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src

)
#    ${rclcpp_INCLUDE_DIRS}
   # ${rmw_INCLUDE_DIRS}
    #${rcl_interfaces_INCLUDE_DIRS}
    #${sensor_msgs_INCLUDE_DIRS}
    #${vision_msgs_INCLUDE_DIRS}
   # ${cv_bridge_INCLUDE_DIRS}
    
# 构建预处理和后处理库
add_library(nn_process SHARED
            src/process/preprocess.cpp
            src/process/postprocess.cpp
)
# 链接库
target_link_libraries(nn_process
    ${OpenCV_LIBS}
)

# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp) 
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)
# yolov8_lib
add_library(yolov8_lib SHARED src/task/yolov8.cpp)
# 链接库
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)


# draw_lib
add_library(draw_lib SHARED src/draw/cv_draw.cpp)
# 链接库
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

# bytetrack_lib
add_library(bytetrack_lib SHARED
  src/bytetrack/BYTETracker.cpp
  src/bytetrack/kalmanFilter.cpp
  src/bytetrack/lapjv.cpp
  src/bytetrack/STrack.cpp
  src/bytetrack/utils.cpp
  )

target_link_libraries(bytetrack_lib Eigen3::Eigen)

# 测试yolov8_video_track
add_executable(object_tracker_node src/yolov8_track_node.cpp)

ament_target_dependencies(object_tracker_node
   rclcpp
   sensor_msgs
   cv_bridge
   vision_msgs
   image_transport
   geometry_msgs
)
# 链接库
target_link_libraries(object_tracker_node
  yolov8_lib
  bytetrack_lib
  #${sensor_msgs_LIBRARIES}
  #${vision_msgs_LIBRARIES}
  #${OpenCV_LIBS}
  #${rcplcpp_LIBRARIES}
  #${rmw_LIBRARIES}
  #${rlc_interfaces_LIBRARIES}
  #cv_bridge::cv_bridge
)
# 安装配置
install(TARGETS nn_process yolov8_lib rknn_engine draw_lib bytetrack_lib
  DESTINATION lib/${PROJECT_NAME}
)



install(TARGETS object_tracker_node
  DESTINATION lib/${PROJECT_NAME}
)
#install(DIRECTORY include/
#  DESTINATION include
#)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
