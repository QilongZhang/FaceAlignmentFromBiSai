/*************************************************************************
	> File Name: FaceAlignment.h
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月05日 星期一 14时43分40秒
 ************************************************************************/

#ifndef _FACEALIGNMENT_H
#define _FACEALIGNMENT_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "cv.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>
#include <utility>


//人脸检测程序输出的矩形
class BoundingBox{
public:
    double start_x;
    double start_y;
    double width;
    double height;
    double centroid_x;
    double centroid_y;
    BoundingBox(){
        start_x = 0;
        start_y = 0;
        width = 0;
        height = 0;
        centroid_x = 0;
        centroid_y = 0;
    }
};

class Fern{
public:
    std::vector<cv::Mat_<double> > Train(const std::vector<std::vector<double> > & candidate_pixel_intensity,
                        const cv::Mat_<double> & covariance,
                        const cv::Mat_<double> & candidate_pixel_locations,
                        const cv::Mat_<int> & nearest_landmark_index,
                        const std::vector<cv::Mat_<double> > & regression_targets,
                        int fern_pixel_num);
    void Write(std::ofstream & fout);

private:
    int fern_pixel_num_;
    int landmark_num_;
    cv::Mat_<int> selected_nearest_landmark_index_;
    cv::Mat_<double> threshold_;
    cv::Mat_<int> selected_pixel_index_;
    cv::Mat_<double> selected_pixel_locations_;
    std::vector<cv::Mat_<double> > bin_output_;
};

class FernCascade{
public:
    std::vector<cv::Mat_<double> > Train(const std::vector<cv::Mat_<uchar> > & images,
                        const std::vector<cv::Mat_<double> > & current_shapes,
                        const std::vector<cv::Mat_<double> > & ground_truth_shapes,
                        const std::vector<BoundingBox> & bounding_box,
                        const cv::Mat_<double> & mean_shape,
                        int second_level_num,
                        int candidate_pixel_num,
                        int fern_pixel_num,
                        int curr_level_num,
                        int first_level_num);
    void Write(std::ofstream & fout);

private:
    std::vector<Fern> ferns_;
    int second_level_num_;
};

class ShapeRegressor{
public:
    ShapeRegressor();
    void Train(const std::vector<cv::Mat_<uchar> > & images,
            const std::vector<cv::Mat_<double> > & ground_truth_shapes,
            const std::vector<BoundingBox> & bounding_box,
            int first_level_num, int second_level_num,
            int candidate_pixel_num, int fern_pixel_num,
            int initial_num);
    void Save(std::string path);
    void Write(std::ofstream & fout);

private:
    int first_level_num_;
    int landmark_num_;
    std::vector<FernCascade> fern_cascades_;
    cv::Mat_<double> mean_shape_;
    std::vector<cv::Mat_<double> > training_shapes_;
    std::vector<BoundingBox> bounding_box_;
};

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> > & shapes,
                const std::vector<BoundingBox> & bounding_box);

cv::Mat_<double> ProjectShape(const cv::Mat_<double> & shape, const BoundingBox & bounding_box);

cv::Mat_<double> ReProjectShape(const cv::Mat_<double> & shape, const BoundingBox & bounding_box);

void SimilarityTransform(const cv::Mat_<double> & shape1, const cv::Mat_<double> & shape2,
                cv::Mat_<double> & rotation, double & scale);

double calculate_covariance(const std::vector<double> & v_1, const std::vector<double> & v_2);

double GetTheta_k(int curr_level_num, double pupils_distance);
#endif
