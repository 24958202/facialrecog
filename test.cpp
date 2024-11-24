/*
 * Program for Facial recognition / Gesture recognition / object recognition
 * Put training image in the folder like:
 
  main folder
       |
  subfolder1, subfolder2,subfolder3...
  
 g++ /Users/dengfengji/ronnieji/lib/new_cvLib/main/test.cpp -o /Users/dengfengji/ronnieji/lib/new_cvLib/main/test -I/Users/dengfengji/ronnieji/lib/new_cvLib/include -I/Users/dengfengji/ronnieji/lib/new_cvLib/src /Users/dengfengji/ronnieji/lib/new_cvLib/src/*.cpp -I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4 -L/opt/homebrew/Cellar/opencv/4.10.0_12/lib -Wl,-rpath,/opt/homebrew/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_video -DOPENCV_VERSION=4.10.0_12 -std=c++20
 * */
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include "cvLib.h"
bool isValidImage(const std::string& img_path){
    if(img_path.empty()){
        return false;
    }
    std::vector<std::string> image_extensions{
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG",
        ".png",
        ".PNG"
    };
    for(const auto& item : image_extensions){
        if(img_path.find(item) != std::string::npos){
            return true;
        }
    }
    return false;
}
void test_image_recognition(){
    std::vector<std::string> testimgs;
    std::string sub_folder_path = "/Users/dengfengji/ronnieji/Kaggle/test"; //"/Users/dengfengji/ronnieji/Kaggle/test";
    for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
        if (entrySubFolder.is_regular_file()) {  
            std::string imgFilePath = entrySubFolder.path().string();  
            if(isValidImage(imgFilePath)){
                testimgs.push_back(imgFilePath);
            }
        }
    }
    std::unordered_map<std::string,return_img_info> results;
    cvLib cvl_j;
    cvl_j.ini_trained_data("/Users/dengfengji/ronnieji/lib/new_cvLib/main/model.dat");//load model.dat before img_recognition
    cvl_j.img_recognition(
        testimgs,
        results
        );
    if(!results.empty()){
        for(const auto& item : results){
            auto it = item.second;
            std::cout << item.first << " is a/an: " << it.objName << '\n';
            std::cout << "Time spent: " << it.timespent << std::endl;
        }
    }
}
int main(){
    /*
            compress all images
    */
     //cvLib cvl_j;
     //cvl_j.img_compress("/Users/dengfengji/ronnieji/Kaggle/archive-2/train",18);
     cvLib cvl_j;
     cvl_j.train_img_occurrences(
         "/Users/dengfengji/ronnieji/Kaggle/gestures/train_test",
         "/Users/dengfengji/ronnieji/lib/new_cvLib/main/model.dat"
     );
     //test_image_recognition();
     cvl_j.ini_trained_data("/Users/dengfengji/ronnieji/lib/new_cvLib/main/model.dat");
     cvl_j.start_recording(0);
}