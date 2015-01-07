/*************************************************************************
	> File Name: ShpaeRegressor.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月05日 星期一 15时04分43秒
 ************************************************************************/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

ShapeRegressor::ShapeRegressor(){
    //...
}

void ShapeRegressor::Train(const vector<Mat_<uchar> > & images,
                        const vector<Mat_<double> > & ground_truth_shapes,
                        const vector<BoundingBox> & bounding_box,
                        int first_level_num, int second_level_num,
                        int candidate_pixel_num, int fern_pixel_num,
                        int initial_num){
    cout<<"Start Training: ShapeRegressor.Train()"<<endl;
    first_level_num_ = first_level_num;
    landmark_num_ = ground_truth_shapes[0].rows;
    training_shapes_ = ground_truth_shapes;
    bounding_box_ = bounding_box;
    
    //训练数据增强，使用多个初始化模型
    vector<Mat_<uchar> > augmented_images;
    vector<BoundingBox> augmented_bounding_box;
    vector<Mat_<double> > augmented_ground_truth_shapes;
    vector<Mat_<double> > current_shapes;

    RNG random_generator(getTickCount());
    for(int i = 0; i < images.size(); ++i){
        for(int j = 0; j < initial_num; ++j){
            int index = 0;
            do{
                index = random_generator.uniform(0, images.size());
            }while(index == i);
            augmented_images.push_back(images[i]);
            augmented_bounding_box.push_back(bounding_box[i]);
            augmented_ground_truth_shapes.push_back(ground_truth_shapes[i]);
            //1.选择其他的图像的ground_truth_shapes当作initial shapes
            //2.把其他图像的ground_truth_shapes投影到相同大小的bounding_box中
            Mat_<double> temp = ground_truth_shapes[index];
            temp = ProjectShape(temp, bounding_box[index]);
            temp = ReProjectShape(temp, bounding_box[i]);
            current_shapes.push_back(temp);
        }    
    }

    //获取所有训练图像的mean shapes，归一化到2×2的bounding_box中
    mean_shape_ = GetMeanShape(ground_truth_shapes, bounding_box);

    //训练一级级联器FernCasacdes
    fern_cascades_.resize(first_level_num);
    vector<Mat_<double> > prediction;
    for(int i = 0; i < first_level_num; ++i){
        cout<<"Training fern cascades: "<<i + 1<<" out of "<<first_level_num<<endl;
        prediction = fern_cascades_[i].Train(augmented_images, current_shapes, augmented_ground_truth_shapes, augmented_bounding_box, mean_shape_, second_level_num, candidate_pixel_num, fern_pixel_num, i + 1, first_level_num);

        //更新下一轮训练的初始shapes
        for(int j = 0; j < prediction.size(); ++j){
            current_shapes[j] = prediction[j] + ProjectShape(current_shapes[j], augmented_bounding_box[j]);
            current_shapes[j] = ReProjectShape(current_shapes[j], augmented_bounding_box[j]);
        }
    }
}

void ShapeRegressor::Write(ofstream & fout){
    fout<<first_level_num_<<endl;
    fout<<mean_shape_.rows<<endl;
    for(int i = 0; i < landmark_num_; ++i){
        fout<<mean_shape_(i, 0)<<" "<<mean_shape_(i, 1)<<" ";
    }
    fout<<endl;

    fout<<training_shapes_.size()<<endl;
    for(int i = 0; i < training_shapes_.size(); ++i){
        fout<<bounding_box_[i].start_x<<" "<<bounding_box_[i].start_y<<" "\
            <<bounding_box_[i].width<<" "<<bounding_box_[i].height<<" "\
            <<bounding_box_[i].centroid_x<<" "<<bounding_box_[i].centroid_y<<endl;
        for(int j = 0; j < training_shapes_[i].rows; ++j){
            fout<<training_shapes_[i](j, 0)<<" "<<training_shapes_[i](j, 1)<<" ";
        }
        fout<<endl;
    }

    for(int i = 0; i < first_level_num_; ++i){
        fern_cascades_[i].Write(fout);
    }
}


void ShapeRegressor::Save(string path){
    cout<<"Saving model..."<<endl;
    ofstream fout;
    fout.open(path.c_str());
    this->Write(fout);
    fout.close();
}
