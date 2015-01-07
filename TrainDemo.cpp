/*************************************************************************
	> File Name: TrainDemo.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月06日 星期二 22时02分54秒
 ************************************************************************/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main(){
    int img_num = 1345;
    int candidate_pixel_num = 400;
    int fern_pixel_num = 5;
    int first_level_num = 20;
    int second_level_num = 500;
    int landmark_num = 29;
    int initial_number = 30;
    vector<Mat_<uchar> > images;

    cout<<"Read images..."<<endl;
    for(int i = 0; i < img_num; ++i){
        string image_name = "./Data/COFW/trainingImages/";
        char to_string[10];
        sprintf(to_string, "%d", i+1);
        image_name = image_name + to_string + ".jpg";
        Mat_<uchar> temp = imread(image_name, 0);
        images.push_back(temp);
    }

    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_box;
    ifstream fin;
    fin.open("./Data/COFW/boundingbox.txt");
    for(int i = 0; i < img_num; ++i){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width /2.0;
        temp.centroid_y = temp.start_y + temp.width /2.0;
        bounding_box.push_back(temp);
    }
    fin.close();

    fin.open("./Data/COFW/keypoints.txt");
    for(int i = 0; i < img_num; ++i){
        Mat_<double> temp(landmark_num, 2);
        for(int j = 0; j < landmark_num; ++j){
            fin>>temp(j, 0);
        }
        for(int j = 0; j < landmark_num; ++j){
            fin>>temp(j, 1);
        }
        ground_truth_shapes.push_back(temp);
    }
    fin.close();

    ShapeRegressor regressor;
    regressor.Train(images, ground_truth_shapes, bounding_box, first_level_num, second_level_num,
                   candidate_pixel_num, fern_pixel_num, initial_number);
    regressor.Save("./Data/MODEL/model_1.txt");
    
    return 0;
}
