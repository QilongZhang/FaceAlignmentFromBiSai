cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)

set(source ./FaceAlignment.h ./Fern.cpp ./FernCascade.cpp ./ShapeRegressor.cpp ./Utils.cpp ./TrainDemo.cpp)
add_executable(TrainDemo.out ${source})
target_link_libraries(TrainDemo.out ${OpenCV_LIBS})

