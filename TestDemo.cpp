/*************************************************************************
	> File Name: TestDemo.cpp
	> Author: QilongZhang
	> Mail: Speknight4534@gmail.com
	> Created Time: 2015年01月10日 星期六 04时38分02秒
 ************************************************************************/

#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int main(){
    vector<Mat_<uchar> > test_images;
    vector<BoundingBox> test_bounding_box;
    int test_img_num = 507;
    int initial_number = 1;
    int landmark_num = 20;
    ifstream fin;

    for(int i = 0; i < test_img_num; ++i){
        string image_name = "./Data/COFW/testImages/";
        char to_string[10];
        sprintf(to_string, "%d", i+1);
        image_name = image_name + to_string + ".jpg";
        Mat_<uchar> temp = imread(image_name, 0);
        test_images.push_back(temp);
    }
    fin.open("./Data/COFW/boundingbox_test.txt");
    for(int i = 0; i < test_img_num; ++i){
        BoundingBox temp;
        fin>>temp.start_x>>temp.start_y>>temp.width>>temp.height;
        temp.centroid_x = temp.start_x + temp.width / 2.0;
        temp.centroid_y = temp.start_y + temp.height / 2.0;
        test_bounding_box.push_back(temp);
    }
    fin.close();

    ShapeRegressor regressor;
    regressor.Load("./Data/MODEL/model_0.txt");
    while(true){
        int index = 0;
        cout<<"Input index:"<<endl;
        cin>>index;

        Mat_<double> current_shape = regressor.Predict(test_images[index], test_bounding_box[index], 1);
        Mat test_image_1 = test_images[index].clone();
        for(int i = 0; i < landmark_num; ++i){
            circle(test_image_1, Point2d(current_shape(i, 0), current_shape(i, 1)), 3, Scalar(255, 0, 0), -1, 8, 0);
        }
        imshow("result", test_image_1);
        waitKey(0);
    }
    return 0;
}
