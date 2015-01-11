/*************************************************************************
	> File Name: Fern.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月06日 星期二 19时40分59秒
 ************************************************************************/

#include "./FaceAlignment.h"
using namespace std;
using namespace cv;

void Fern::Train(const vector<vector<double> > & candidate_pixel_intensity,
                    const Mat_<double> & covariance,
                    const Mat_<double> & candidate_pixel_locations,
                    const Mat_<int> & nearest_landmark_index,
                    const vector<Mat_<double> > & ground_truth_shapes,
                    vector<Mat_<double> > & current_shapes,
                    const vector<BoundingBox> & bounding_box,
                    const Mat_<double> & mean_shape,
                    vector<Mat_<double> > & regression_targets,
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

    //计算每个bin的bin_output_detTheta_;
    bin_output_detTheta_.resize(bin_num);
    for(int i = 0; i < bin_num; ++i){
        int bin_size = shapes_in_bin[i].size();
        double temp = 0;
        double scale;
        Mat_<double> rotation;
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            SimilarityTransform(ground_truth_shapes[index], current_shapes[index], rotation, scale);
            temp += asin(rotation(0,1)) * 180 / PI;
        }
        if(bin_size == 0){
            bin_output_detTheta_[i] = temp;
            continue;
        }
        temp = (1.0 / ((1.0 + 1000.0 / bin_size) * bin_size)) * temp;
        bin_output_detTheta_[i] = temp;
        rotation(0, 0) = cos(temp * PI / 180);
        rotation(0, 1) = sin(temp * PI / 180);
        rotation(1, 0) = -sin(temp * PI / 180);
        rotation(1, 1) = cos(temp * PI / 180);
        //旋转角度，更改旋转后的regression_targets
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            current_shapes[index] = current_shapes[index] * rotation;
            regression_targets[index] = ProjectShape(ground_truth_shapes[index], bounding_box[index]) - ProjectShape(current_shapes[index], bounding_box[index]);
            Mat_<double> temp_rotation;
            double scale;
            //normalize regression targets
            SimilarityTransform(mean_shape, ProjectShape(current_shapes[index], bounding_box[index]), temp_rotation, scale);
            regression_targets[index] = scale * regression_targets[index] * temp_rotation;
        }
    }

    //计算每个bin的bin_output_detShape_;
    bin_output_detShape_.resize(bin_num);
    for(int i = 0; i < bin_num; ++i){
        Mat_<double> temp = Mat::zeros(landmark_num_, 2, CV_64FC1);
        int bin_size = shapes_in_bin[i].size();
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            temp += regression_targets[index];
        }
        if(bin_size == 0){
            bin_output_detShape_[i] = temp;
            continue;
        }
        temp = (1.0 / ((1.0 + 1000.0 / bin_size) * bin_size)) * temp;
        bin_output_detShape_[i] = temp;
        for(int j = 0; j < bin_size; ++j){
            int index = shapes_in_bin[i][j];
            regression_targets[index] -= temp;
            current_shapes[index] = temp + ProjectShape(current_shapes[index], bounding_box[index]);
            current_shapes[index] = ReProjectShape(current_shapes[index], bounding_box[index]);
        }
    }
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
    for(int i = 0; i < bin_output_detShape_.size(); ++i){
        fout<<bin_output_detTheta_[i]<<endl;
        for(int j = 0; j < bin_output_detShape_[i].rows; ++j){
            fout<<bin_output_detShape_[i](j, 0)<<" "<<bin_output_detShape_[i](j, 1)<<" ";
        }
        fout<<endl;
    }
}

void Fern::Read(ifstream & fin){
    fin>>fern_pixel_num_;
    fin>>landmark_num_;
    selected_nearest_landmark_index_.create(fern_pixel_num_, 2);
    selected_pixel_locations_.create(fern_pixel_num_, 4);
    threshold_.create(fern_pixel_num_, 1);
    
    for(int i = 0; i < fern_pixel_num_; ++i){
        fin>>selected_pixel_locations_(i, 0)>>selected_pixel_locations_(i, 1)
            >>selected_pixel_locations_(i, 2)>>selected_pixel_locations_(i, 3);
        fin>>selected_nearest_landmark_index_(i, 0)>>selected_nearest_landmark_index_(i, 1);
        fin>>threshold_(i);
    }

    int bin_num = pow(2.0, fern_pixel_num_);
    bin_output_detTheta_.resize(bin_num);
    for(int i = 0; i < bin_num; ++i){
        Mat_<double> temp(landmark_num_, 2);
        fin>>bin_output_detTheta_[i];
        for(int j = 0; j < landmark_num_; ++j){
            fin>>temp(j, 0)>>temp(j, 1);
        }
        bin_output_detShape_.push_back(temp);
    }
}

void Fern::Predict(const Mat_<uchar> & image, const BoundingBox & bounding_box, const Mat_<double> & mean_shape, 
        Mat_<double> & shape){
    Mat_<double> rotation;
    double scale;
    SimilarityTransform(ProjectShape(shape, bounding_box), mean_shape, rotation, scale);
    int index = 0;
    for(int i = 0; i < fern_pixel_num_; ++i){
        int nearest_landmark_index_1 = selected_nearest_landmark_index_(i, 0);
        int nearest_landmark_index_2 = selected_nearest_landmark_index_(i, 1);
       
        double x = selected_pixel_locations_(i, 0);
        double y = selected_pixel_locations_(i, 1);
        double project_x = scale * (x * rotation(0, 0) + y * rotation(1, 0)) * bounding_box.width / 2.0 
            + shape(nearest_landmark_index_1, 0);
        double project_y = scale * (x * rotation(0, 1) + y * rotation(1 ,1)) * bounding_box.height / 2.0 
            + shape(nearest_landmark_index_1, 1);
        project_x = std::max(0.0, std::min((double)project_x, image.cols - 1.0));
        project_y = std::max(0.0, std::min((double)project_y, image.rows - 1.0));
        double intensity_1 = (int)(image((int)project_y, (int)project_x));

        x = selected_pixel_locations_(i, 2);
        y = selected_pixel_locations_(i, 3);
        project_x = scale * (x * rotation(0, 0) + y * rotation(1, 0)) * bounding_box.width / 2.0 
            + shape(nearest_landmark_index_2, 0);
        project_y = scale * (x * rotation(0, 1) + y * rotation(1, 1)) * bounding_box.height / 2.0
            + shape(nearest_landmark_index_2, 1);
        project_x = std::max(0.0, std::min((double)project_x, image.cols - 1.0));
        project_y = std::max(0.0, std::min((double)project_y, image.rows - 1.0));
        double intensity_2 = (int)(image((int)project_y, (int)project_x));
    
        if(intensity_1 - intensity_2 >= threshold_(i)){
            index += pow(2, i);
        }
    }

    shape = ProjectShape(shape, bounding_box);
    rotation(0, 0) = cos(bin_output_detTheta_[index] * PI / 180);
    rotation(0, 1) = sin(bin_output_detTheta_[index] * PI / 180);
    rotation(1, 0) = -sin(bin_output_detTheta_[index] * PI / 180);
    rotation(1, 1) = cos(bin_output_detTheta_[index] * PI / 180);
    shape = shape * rotation;
    shape += bin_output_detShape_[index];
    ReProjectShape(shape, bounding_box);
}
