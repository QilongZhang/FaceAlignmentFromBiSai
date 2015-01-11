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
    for(int i = 0; i < first_level_num; ++i){
        cout<<"Training fern cascades: "<<i + 1<<" out of "<<first_level_num<<endl;
        fern_cascades_[i].Train(augmented_images, 
                                current_shapes, 
                                augmented_ground_truth_shapes,
                                augmented_bounding_box, 
                                mean_shape_, 
                                second_level_num, 
                                candidate_pixel_num, 
                                fern_pixel_num, 
                                i + 1, 
                                first_level_num);
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

void ShapeRegressor::Read(ifstream & fin){
    fin>>first_level_num_;
    fin>>landmark_num_;
    mean_shape_ = Mat::zeros(landmark_num_, 2, CV_64FC1);
    for(int i = 0; i < landmark_num_; ++i){
        fin>>mean_shape_(i, 0)>>mean_shape_(i, 1);
    }
    
    int training_num;
    fin>>training_num;
    training_shapes_.resize(training_num);
    bounding_box_.resize(training_num);

    for(int i = 0; i < training_num; ++i){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height>>temp.centroid_x>>temp.centroid_y;
        bounding_box_[i] = temp;
        
        Mat_<double> temp1(landmark_num_, 2);
        for(int j = 0; j < landmark_num_; ++j){
            fin>>temp1(j, 0)>>temp1(j, 1);
        }
        training_shapes_[i] = temp1;
    }

    fern_cascades_.resize(first_level_num_);
    for(int i = 0; i < first_level_num_; ++i){
        fern_cascades_[i].Read(fin);
    }
}

void ShapeRegressor::Load(string path){
    cout<<"Loading model..."<<endl;
    ifstream fin(path.c_str());
    if(!fin){
        cout<<"ERROR"<<endl;
        return ;
    }
    Read(fin);
    fin.close();
    cout<<"Model loaded successfully,,,"<<endl;
}

Mat_<double> ShapeRegressor::Predict(const Mat_<uchar> & image, const BoundingBox & bounding_box, int initial_num){
    //产生多个初始化形状
    Mat_<double> result = Mat::zeros(landmark_num_, 2, CV_64FC1);
    RNG random_generator(getTickCount());
    for(int i = 0; i < initial_num; ++i){
        int index = random_generator.uniform(0, training_shapes_.size());
        Mat_<double> current_shape = training_shapes_[index];
        BoundingBox current_bounding_box = bounding_box_[index];
        current_shape = ProjectShape(current_shape, current_bounding_box);
        current_shape = ReProjectShape(current_shape, bounding_box);
        for(int j = 0; j < first_level_num_; ++j){
            fern_cascades_[j].Predict(image, bounding_box, mean_shape_, current_shape);
        }
        result = result + current_shape;
    }
    return 1.0 / initial_num * result;
}
