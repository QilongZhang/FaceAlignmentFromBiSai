/*************************************************************************
	> File Name: Fern.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月06日 星期二 19时40分59秒
 ************************************************************************/

#include "./FaceAlignment.h"
using namespace std;
using namespace cv;

vector<Mat_<double> > Fern::Train(const vector<vector<double> > & candidate_pixel_intensity,
                    const Mat_<double> & covariance,
                    const Mat_<double> & candidate_pixel_locations,
                    const Mat_<int> & nearest_landmark_index,
                    const vector<Mat_<double> > & regression_targets,
                    int fern_pixel_num){  
    fern_pixel_num_ = fern_pixel_num;
    landmark_num_ = regression_targets[0].rows;
    selected_pixel_index_.create(fern_pixel_num, 2);
    selected_pixel_locations_.create(fern_pixel_num, 4);
    selected_nearest_landmark_index_.create(fern_pixel_num, 2);
    int candidate_pixel_num = candidate_pixel_locations.rows;
    threshold_.create(fern_pixel_num, 1);

    //选择fern_pixel_num对特征
    RNG random_generator(getTickCount());
    for(int i = 0; i < fern_pixel_num; ++i){
        //生成一个随机的方向基
        Mat_<double> random_direction(landmark_num_, 2);
        random_generator.fill(random_direction, RNG::UNIFORM, -1.1, 1.1);
        normalize(random_direction, random_direction);
        vector<double> projection_result(regression_targets.size(), 0);
        //把regression targets向该方向基投影
        for(int j = 0; j < regression_targets.size(); ++j){
            double temp = 0;
            temp = sum(regression_targets[j].mul(random_direction))[0];
            projection_result[j] = temp;
        }
        Mat_<double> covariance_projection_density(candidate_pixel_num, 1);
        for(int j = 0; j < candidate_pixel_num; ++j){
            covariance_projection_density(j) = calculate_covariance(projection_result, candidate_pixel_intensity[j]);
        }

        //找到一对最相关的特征
        double max_correlation = -1;
        int max_pixel_index_1 = 0;
        int max_pixel_index_2 = 0;
        for(int j = 0; j < candidate_pixel_num; ++j){
            for(int k = 0; k < candidate_pixel_num; ++k){
                double temp1 = covariance(j, j) + covariance(k, k) - 2 * covariance(j ,k);
                if(abs(temp1) < 1e-10){
                    continue;
                }
                bool flag = false;
                for(int p = 0; p < i; ++p){
                    if(j == selected_pixel_index_(p, 0) && k == selected_pixel_index_(p, 1)){
                        flag = true;
                        break;
                    }
                    else if(j == selected_pixel_index_(p, 1) && k == selected_pixel_index_(p, 0)){
                        flag = true;
                        break;
                    }
                }
                if(flag){
                    continue;
                }
                double temp = (covariance_projection_density(j) - covariance_projection_density(k)) / sqrt(temp1);
                if(abs(temp) > max_correlation){
                    max_correlation = temp;
                    max_pixel_index_1 = j;
                    max_pixel_index_2 = k;
                }
            }
        }

        selected_pixel_index_(i, 0) = max_pixel_index_1;
        selected_pixel_index_(i, 1) = max_pixel_index_2;
        selected_pixel_locations_(i, 0) = candidate_pixel_locations(max_pixel_index_1, 0);
        selected_pixel_locations_(i, 1) = candidate_pixel_locations(max_pixel_index_1, 1);
        selected_pixel_locations_(i, 2) = candidate_pixel_locations(max_pixel_index_2, 0);
        selected_pixel_locations_(i, 3) = candidate_pixel_locations(max_pixel_index_2, 1);
        selected_nearest_landmark_index_(i, 0) = nearest_landmark_index(max_pixel_index_1);
        selected_nearest_landmark_index_(i, 1) = nearest_landmark_index(max_pixel_index_2);

        //计算该对特征在测试时的阈值
        double max_diff = -1;
        for(int j = 0; j < candidate_pixel_intensity[max_pixel_index_1].size(); ++j){
            double temp = candidate_pixel_intensity[max_pixel_index_1][j] \
                          - candidate_pixel_intensity[max_pixel_index_2][j];
            if(abs(temp) > max_diff){
                max_diff = abs(temp);
            }
        }

        threshold_(i) = random_generator.uniform(-0.2 * max_diff, 0.2 * max_diff);
    }
    
    //计算落入每个bin对应的图像编号
    vector<vector<int> > shapes_in_bin;
    int bin_num = pow(2.0, fern_pixel_num);
    shapes_in_bin.resize(bin_num);
    for(int i = 0; i < regression_targets.size(); ++i){
        int index = 0;
        for(int j = 0; j < fern_pixel_num; ++j){
            double density_1 = candidate_pixel_intensity[selected_pixel_index_(j, 0)][i];
            double density_2 = candidate_pixel_intensity[selected_pixel_index_(j ,1)][i];
            if(density_1 - density_2 >= threshold_(j)){
                index = index + pow(2.0, j);
            }
        }
        shapes_in_bin[index].push_back(i);
    }

    //计算每个bin的输出
    vector<Mat_<double> > prediction;
    prediction.resize(regression_targets.size());
    bin_output_.resize(bin_num);
    for(int i = 0; i < bin_num; ++i){
        Mat_<double> temp = Mat::zeros(landmark_num_, 2, CV_64FC1);
        int bin_size = shapes_in_bin[i].size();
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            temp += regression_targets[index];
        }
        if(bin_size == 0){
            bin_output_[i] = temp;
            continue;
        }
        temp = (1.0 / ((1.0 + 1000.0 / bin_size) * bin_size)) * temp;
        bin_output_[i] = temp;
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            prediction[index] = temp;
        }
    }
    return prediction;
}

void Fern::Write(ofstream & fout){
    fout<<fern_pixel_num_<<endl;
    fout<<landmark_num_<<endl;
    for(int i = 0; i < fern_pixel_num_; ++i){
        fout<<selected_pixel_locations_(i, 0)<<" "<<selected_pixel_locations_(i, 1)<<" " \
            <<selected_pixel_locations_(i, 2)<<" "<<selected_pixel_locations_(i, 3)<<" "<<endl;
        fout<<selected_nearest_landmark_index_(i, 0)<<endl;
        fout<<selected_nearest_landmark_index_(i, 1)<<endl;
        fout<<threshold_(i)<<endl;
    }
    for(int i = 0; i < bin_output_.size(); ++i){
        for(int j = 0; j < bin_output_[i].rows; ++j){
            fout<<bin_output_[i](j, 0)<<" "<<bin_output_[i](j, 1)<<" ";
        }
        fout<<endl;
    }
}

