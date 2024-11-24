#ifndef CVLIB_H
#define CVLIB_H
#include <iostream> 
#include <vector>
#include <unordered_map>
#include <map>
#include <opencv2/opencv.hpp>  
#include <functional>
#include <stdint.h>
#include <chrono>
#ifdef __cplusplus
extern "C" {
#endif
const uint8_t C_0 = 0;
const uint8_t C_1 = 1;
const uint8_t C_2 = 2;
const uint8_t C_3 = 3;
struct return_img_info{
    std::string objName;
    std::chrono::duration<double> timespent;
};
class cvLib{
    public:
        class subclasses;
        /*
             Function to compress images 
             para1: input image folder with different catalog subfolders 
             para2: image quality
         */
        void img_compress(const std::string&,int);
        /*
            para1: trained folder path
            main folder
                |
            catalog1,catalog2,catalog3...
            para2: output model file path: model.dat
        */
        void train_img_occurrences(
            const std::string&,
            const std::string&
        );
        /*
            load model file
        */
        void loadModel_keypoint(std::unordered_map<std::string, std::vector<cv::Mat>>&, const std::string&);
        /*
         * Function to load trained data
         * para1: train data model.dat file path
         */
        void ini_trained_data(const std::string&);
        /*
            para1: input test image path dataset std::vector<std::string>
            para2: output image recognition result
        */
        void img_recognition(const std::vector<std::string>&,std::unordered_map<std::string,return_img_info>&);
        /*
         Function to recognize objs in video 
         Facial recognition
         para1: webcam index number default : 0
        */
        void start_recording(unsigned int);
    private:
        std::unordered_map<std::string, std::vector<cv::Mat>> trained_dataset;
        void checkExistingGestures(const cv::Mat&, std::string&);
        bool checkExistingFace(const std::string&, const cv::Mat&);
        void onFacesDetected(const std::vector<cv::Rect>&, cv::Mat&, const std::string&);
};

#ifdef __cplusplus
}
#endif
#endif