/*************************************************************************
	> File Name: Utils.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月05日 星期一 16时00分42秒
 ************************************************************************/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

Mat_<double> ProjectShape(const Mat_<double> & shape, const BoundingBox & bounding_box){
    Mat_<double> result(shape.rows, 2);
    for(int i = 0; i < shape.rows; ++i){
        result(i, 0) = (shape(i, 0) - bounding_box.centroid_x) / (bounding_box.width / 2.0);
        result(i, 1) = (shape(i, 1) - bounding_box.centroid_y) / (bounding_box.height / 2.0);
    }
    return result;
}

Mat_<double> ReProjectShape(const Mat_<double> & shape, const BoundingBox & bounding_box){
    Mat_<double> result(shape.rows, 2);
    for(int i = 0; i < shape.rows; ++i){
        result(i, 0) = (shape(i, 0) * bounding_box.width / 2.0 + bounding_box.centroid_x);
        result(i, 1) = (shape(i, 1) * bounding_box.height / 2.0 + bounding_box.centroid_y);
    }
    return result;
}

Mat_<double> GetMeanShape(const vector<Mat_<double> > & shapes, const vector<BoundingBox> & bounding_box){
    Mat_<double> result = Mat::zeros(shapes[0].rows, 2, CV_64FC1);
    for(int i = 0; i < shapes.size(); ++i){
        result += ProjectShape(shapes[i], bounding_box[i]);
    }
    result = 1.0 / shapes.size() * result;
    return result;
}

void SimilarityTransform(const Mat_<double> & shape1, const Mat_<double> & shape2,
        Mat_<double> & rotation, double & scale){
    rotation = Mat::zeros(2, 2, CV_64FC1);
    scale = 0;

    //获取shape1和shape2的形心
    double centroid_x_1 = 0;
    double centroid_y_1 = 0;
    double centroid_x_2 = 0;
    double centroid_y_2 = 0;
    for(int i = 0; i < shape1.rows; ++i){
        centroid_x_1 += shape1(i, 0);
        centroid_y_1 += shape1(i, 1);
        centroid_x_2 += shape2(i, 0);
        centroid_y_2 += shape2(i, 1);
    }
    centroid_x_1 /= shape1.rows;
    centroid_y_1 /= shape1.rows;
    centroid_x_2 /= shape2.rows;
    centroid_y_2 /= shape2.rows;

    Mat_<double> temp1 = shape1.clone();
    Mat_<double> temp2 = shape2.clone();
    //分别以质心作为原点建立坐标轴
    for(int i = 0; i < shape1.rows; ++i){
        temp1(i, 0) -= centroid_x_1;
        temp1(i, 1) -= centroid_y_1;
        temp2(i, 0) -= centroid_x_2;
        temp2(i, 1) -= centroid_y_2;
    }
    
    //Uniform scaling
    double s1 = 0;
    double s2 = 0;
    for(int i = 0; i < temp1.rows; ++i){
        s1 += pow(temp1(i, 0), 2.0);
        s1 += pow(temp1(i ,1), 2.0);
        s2 += pow(temp2(i, 0), 2.0);
        s2 += pow(temp2(i, 1), 2.0);
    }
    s1 = sqrt(s1/temp1.rows);
    s2 = sqrt(s2/temp2.rows);
    temp1 = 1.0 / s1 * temp1;
    temp2 = 1.0 / s2 * temp2;
    scale = s1 / s2;

    //Rotation
    double hx = 0;
    double vy = 0;
    for(int i = 0; i < temp1.rows; ++i){
        vy += temp2(i, 0) * temp1(i, 1) - temp2(i, 1) * temp1(i, 0);
        hx += temp2(i, 0) * temp1(i, 0) + temp2(i, 1) * temp1(i, 1);
    }
    double hv = sqrt(pow(hx, 2) + pow(vy, 2));
    double sin_theta = vy / hv;
    double cos_theta = hx / hv;
    rotation(0, 0) = cos_theta;
    rotation(0, 1) = sin_theta;
    rotation(1, 0) = -sin_theta;
    rotation(1, 1) = cos_theta;
}

double GetTheta_k(int curr_level_num, double pupils_distance){
    if(curr_level_num == 1)
        return 0.5 * pupils_distance;
    else if(curr_level_num == 2)
        return 0.45 * pupils_distance;
    else if(curr_level_num == 3)
        return 0.4 * pupils_distance;
    else if(curr_level_num == 4 || curr_level_num == 5)
        return 0.35 * pupils_distance;
    else if(curr_level_num = 6 || curr_level_num == 7)
        return 0.3 * pupils_distance;
    else if(curr_level_num == 8 || curr_level_num == 9)
        return 0.2 * pupils_distance;
    else 
        return 0.1 * pupils_distance;
}

double calculate_covariance(const vector<double> & v_1, const vector<double> & v_2){
    Mat_<double> v1(v_1);
    Mat_<double> v2(v_2);
    double mean_1 = mean(v1)[0];
    double mean_2 = mean(v2)[0];
    v1 = v1 - mean_1;
    v2 = v2 - mean_2;
    return mean(v1.mul(v2))[0];
}


