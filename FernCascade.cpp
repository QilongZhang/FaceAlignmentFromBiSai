/*************************************************************************
	> File Name: FernCascade.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月06日 星期二 09时37分11秒
 ************************************************************************/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

vector<Mat_<double> > FernCascade::Train(const vector<Mat_<uchar> > & images,
                const vector<Mat_<double> > & current_shapes,
                const vector<Mat_<double> > & ground_truth_shapes,
                const vector<BoundingBox> & bounding_box,
                const Mat_<double> & mean_shape,
                int second_level_num,
                int candidate_pixel_num,
                int fern_pixel_num,
                int curr_level_num,
                int first_level_num){ 
    Mat_<double> candidate_pixel_locations(candidate_pixel_num, 2);
    Mat_<int> nearest_landmark_index(candidate_pixel_num, 1);
    vector<Mat_<double> > regression_targets;
    RNG random_generator(getTickCount());
    second_level_num_ = second_level_num;

    //获取regression targets
    regression_targets.resize(current_shapes.size());
    for(int i = 0; i < current_shapes.size(); ++i){
        regression_targets[i] = ProjectShape(ground_truth_shapes[i], bounding_box[i]) - ProjectShape(current_shapes[i], bounding_box[i]);
        Mat_<double> rotation;
        double scale;
        //normalize regression targets
        SimilarityTransform(mean_shape, ProjectShape(current_shapes[i], bounding_box[i]), rotation, scale);
        regression_targets[i] = scale * regression_targets[i] * rotation;
    }

    //获取候选的shape-indexed features
    double pupils_distance = sqrt(pow(mean_shape(17,0) - mean_shape(16, 0), 2) + pow(mean_shape(17, 1) - mean_shape(16, 1), 2));
    //论文给的theta_k是个定值，0.3倍的眼球距离
    double theta_k = pupils_distance * 0.3;
    //是否可以随着curr_level_num变化
    //double theta_k = GetTheta_k(curr_level_num, pupils_distance);
    for(int i = 0; i < candidate_pixel_num; ++i){
        int min_index = random_generator.uniform(0, mean_shape.rows);
        double det_x = random_generator.uniform(-theta_k, theta_k);
        double det_y = random_generator.uniform(-theta_k, theta_k);
        if(pow(det_x, 2.0) + pow(det_y, 2.0) > pow(theta_k, 2.0)){
           --i;
           continue;
        }
        double min_dist = pow(det_x, 2.0) + pow(det_y, 2.0);
        double x = det_x + mean_shape(min_index, 0);
        double y = det_y + mean_shape(min_index, 1);
        //找到最近的landmarks indexed
        for(int j = 0; j < mean_shape.rows; ++j){
            double temp = pow(mean_shape(j, 0) - x, 2.0) + pow(mean_shape(j, 1) - y, 2.0);
            if(temp < min_dist){
                min_dist = temp;
                min_index = j;
            }
        }
        candidate_pixel_locations(i, 0) = x - mean_shape(min_index, 0);
        candidate_pixel_locations(i, 1) = y - mean_shape(min_index, 1);
        nearest_landmark_index(i) = min_index;
    }

    //获取每张图片对应的shape-indexed point 的灰度值
    vector<vector<double> > densities;
    densities.resize(candidate_pixel_num);
    for(int i = 0; i < images.size(); ++i){
        Mat_<double> rotation;
        double scale;
        Mat_<double> temp = ProjectShape(current_shapes[i], bounding_box[i]);
        SimilarityTransform(temp, mean_shape, rotation, scale);
        for(int j = 0; j < candidate_pixel_num; ++j){
            double project_x = rotation(0, 0) * candidate_pixel_locations(j, 0)
                + rotation(1, 0) * candidate_pixel_locations(j, 1);
            double project_y = rotation(0, 1) * candidate_pixel_locations(j ,0)
                + rotation(1, 1) * candidate_pixel_locations(j ,1);
            project_x = scale * project_x * bounding_box[i].width / 2.0;
            project_y = scale * project_y * bounding_box[i].height / 2.0;
            int index = nearest_landmark_index(j);
            int real_x = project_x + current_shapes[i](index, 0);
            int real_y = project_y + current_shapes[i](index, 1);
            real_x = std::max(0.0, std::min((double)real_x, images[i].cols - 1.0));
            real_y = std::max(0.0, std::min((double)real_y, images[i].rows - 1.0));
            densities[j].push_back((double)images[i](real_y, real_x));
        }
    }

    //计算densities中行向量之间的协方差
    Mat_<double> covariance(candidate_pixel_num, candidate_pixel_num);
    for(int i = 0; i < candidate_pixel_num; ++i){
        for(int j = i; j < candidate_pixel_num; ++j){
            double correlation_result = calculate_covariance(densities[i], densities[j]);
            covariance(i, j) = correlation_result;
            covariance(j, i) = correlation_result;
        }
    }

    //训练二级分类器 Ferns
    vector<Mat_<double> > prediction;
    prediction.resize(regression_targets.size());
    for(int i = 0; i < regression_targets.size(); ++i){
        prediction[i] = Mat::zeros(mean_shape.rows, 2, CV_64FC1);
    }
    ferns_.resize(second_level_num);
    clock_t t = clock();
    for(int i = 0; i < second_level_num; ++i){
        vector<Mat_<double> > temp = ferns_[i].Train(densities, covariance, candidate_pixel_locations,
                nearest_landmark_index, regression_targets, fern_pixel_num);
        //update regression targets
        for(int j = 0; j < temp.size(); ++j){
            prediction[j] += temp[j];
            regression_targets[j] -= temp[j];
        }
        if((i + 1) % 50 == 0){
            cout<<"Fern cascades: " << curr_level_num << " out of " << first_level_num<<";"<<endl;
            cout<<"Ferns: "<< i + 1 << " out of "<<second_level_num<<endl;
            double remaining_level_num = (first_level_num - curr_level_num) * 500 + second_level_num - i;
            double time_remaining = 0.02 * double(clock() - t) / CLOCKS_PER_SEC * remaining_level_num;
            cout<<"Expected remaining time: "\
                <<(int)time_remaining / 60 << " min "<<(int)time_remaining % 60 <<" s"<<endl;
            t = clock();
        }
    }

    for(int i = 0; i < prediction.size(); ++i){
        Mat_<double> rotation;
        double scale;
        SimilarityTransform(ProjectShape(current_shapes[i], bounding_box[i]), mean_shape, rotation, scale);
        prediction[i] = scale * prediction[i] * rotation;
    }
    return prediction;
}

void FernCascade::Write(ofstream & fout){
    fout<<second_level_num_<<endl;
    for(int i = 0; i < second_level_num_; ++i){
        ferns_[i].Write(fout);
    }
}
