/*
    c++20 lib for using opencv
    Dependencies:
    opencv,tesseract,sdl2,sdl_image,boost
*/
#include <opencv2/opencv.hpp>  
#include <opencv2/features2d.hpp> 
#include <opencv2/video.hpp> 
#include "authorinfo/author_info.h" 
#include <vector> 
#include <tuple> 
#include <queue>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cmath> // For std::abs
#include <map>
#include <set>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>//for std::max
#include <numeric>
#include <ranges> //std::views
#include <cstdint> 
#include <functional>
#include <cstdlib>
#include <unordered_set>
#include <iterator> 
#include <utility>        // For std::pair  
#include <execution> // for parallel execution policies (C++17)
#include <jpeglib.h>
#include <stdlib.h>
#include "cvLib.h"
const unsigned int MAX_FEATURES = 1000;   // Max number of features to detect
const float RATIO_THRESH = 0.95f;          // Ratio threshold for matching
const unsigned int DE_THRESHOLD = 10;      // Min matches to consider an object as existing 10
const unsigned int ReSIZE_IMG_WIDTH = 100;
const unsigned int ReSIZE_IMG_HEIGHT = 100;
unsigned int faceCount = 0; 
class cvLib::subclasses{
    public:
        /*
            check if a file extension contains image extension
        */
        bool isValidImage(const std::string&);
        /*
            Function to compress images 
            para1: input image file path
            para2: output image file path
            para3: image quality 0-100
         */
        void compressJPEG(const std::string&, const std::string&, int);
        /*
         Function to resize and blur input image
         para1: image path 
         para2: output image
         */
        void preprocessImg(const std::string&,cv::Mat&);
        /*
         para1: input image cv::Mat
         para2: output descriptors
         */
        std::vector<cv::KeyPoint> extractORBFeatures(const cv::Mat&, cv::Mat&);
        void saveModel_keypoint(const std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename);
        /*
          Function to convert cv::KeyPoint to descriptors
        */
        cv::Mat computeDescriptors(std::vector<cv::KeyPoint>&);
        /*
            Function to compare two images 
            para1: img1 file path
            para2: img2 file path
        */
        bool img1_img2_are_matched(const std::string&, const std::string&);
        
};
bool cvLib::subclasses::isValidImage(const std::string& img_path){
    if(img_path.empty()){
        return false;
    }
    std::vector<std::string> image_extensions{
        ".jpg",
        ".JPG",
        ".jpeg",
        ".JPEG"
        //".png",
        //".PNG"
    };
    for(const auto& item : image_extensions){
        if(img_path.find(item) != std::string::npos){
            return true;
        }
    }
    return false;
}
void cvLib::subclasses::compressJPEG(const std::string& inputFilename, const std::string& outputFilename, int quality) {
    if (inputFilename.empty() || outputFilename.empty()) {
        throw std::invalid_argument("Input and output filenames must not be empty.");
    }
    // Create a JPEG compression struct and error handler
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    // Open the input file
    FILE* infile = fopen(inputFilename.c_str(), "rb");
    if (!infile) {
        jpeg_destroy_compress(&cinfo);
        throw std::runtime_error("Unable to open input file: " + inputFilename);
    }
    // Initialize JPEG decompression
    jpeg_decompress_struct dinfo;
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);
    jpeg_stdio_src(&dinfo, infile);
    if (jpeg_read_header(&dinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&dinfo);
        fclose(infile);
        throw std::runtime_error("Failed to read JPEG header from " + inputFilename);
    }
    jpeg_start_decompress(&dinfo);
    // Get image properties
    int width = dinfo.output_width;
    int height = dinfo.output_height;
    int numChannels = dinfo.num_components;
    // Allocate memory for the image data
    unsigned char* buffer = new unsigned char[width * height * numChannels];
    while (dinfo.output_scanline < height) {
        unsigned char* row_pointer = buffer + dinfo.output_scanline * width * numChannels;
        if (jpeg_read_scanlines(&dinfo, &row_pointer, 1) != 1) {
            delete[] buffer;
            jpeg_destroy_decompress(&dinfo);
            fclose(infile);
            throw std::runtime_error("Failed to read scanlines from " + inputFilename);
        }
    }
    // Finish decompression and close the input file
    jpeg_finish_decompress(&dinfo);
    fclose(infile);
    jpeg_destroy_decompress(&dinfo);
    // Set up compression
    FILE* outfile = fopen(outputFilename.c_str(), "wb");
    if (!outfile) {
        delete[] buffer;
        throw std::runtime_error("Unable to open output file: " + outputFilename);
    }
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = numChannels;
    // Comprehensive color space handling
    switch (numChannels) {
        case 1:
            cinfo.in_color_space = JCS_GRAYSCALE;
            break;
        case 3:
            cinfo.in_color_space = JCS_RGB; // Assuming RGB
            break;
        case 4:
            cinfo.in_color_space = JCS_CMYK; // Assuming CMYK, if supported
            break;
        default:
            delete[] buffer;
            fclose(outfile);
            throw std::invalid_argument("Unsupported number of channels: " + std::to_string(numChannels));
    }
    jpeg_set_defaults(&cinfo);
    // Check quality value
    if (quality < 0 || quality > 100) {
        delete[] buffer;
        fclose(outfile);
        throw std::invalid_argument("Quality must be between 0 and 100.");
    }
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_stdio_dest(&cinfo, outfile);
    // Start compression
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char* row_pointer = buffer + cinfo.next_scanline * width * numChannels;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    // Finish compression and clean up
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    delete[] buffer;
    std::cout << "JPEG image compressed and saved to " << outputFilename << std::endl;
}
void cvLib::subclasses::preprocessImg(const std::string& img_path, cv::Mat& outImg){
    if (img_path.empty()) {
        std::cerr << "Error: img_path is empty." << std::endl;
        return;
    }
    try {
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Error: Image not loaded correctly from path: " << img_path << std::endl;
            return; // Handle the error appropriately
        }
        cv::Mat resizeImg;
        cv::resize(img, resizeImg, cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT));
        cv::Mat img_gray;
        cv::cvtColor(resizeImg, img_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(img_gray, outImg, cv::Size(5, 5), 0);
    } catch (const cv::Exception& ex) {
        std::cerr << "OpenCV Error: " << ex.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Standard Error: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "cvLib::subclasses::preprocessImg Unknown errors" << std::endl;
    }
}
std::vector<cv::KeyPoint> cvLib::subclasses::extractORBFeatures(const cv::Mat& img, cv::Mat& descriptors){
    std::vector<cv::KeyPoint> keypoints_input;
    if(img.empty()){
        return keypoints_input;
    }
    try{
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(MAX_FEATURES);
        detector->detectAndCompute(img, cv::noArray(), keypoints_input, descriptors);
    }
    catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
    }
    catch(...){
        std::cerr << "cvLib::subclasses::extractORBFeatures Unknown errors" << std::endl;
    }
    return keypoints_input;
}
void cvLib::subclasses::saveModel_keypoint(const std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename){
    if (filename.empty()) {
        return;
    }
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Unable to open file for writing.");
    }
    try {
        size_t mapSize = featureMap.size();
        ofs.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));
        for (const auto& [className, images] : featureMap) {
            size_t keySize = className.size();
            ofs.write(reinterpret_cast<const char*>(&keySize), sizeof(keySize));
            ofs.write(className.data(), keySize);
            size_t imageCount = images.size();
            ofs.write(reinterpret_cast<const char*>(&imageCount), sizeof(imageCount));
            for (const auto& img : images) {
                // Write image dimensions
                int rows = img.rows;
                int cols = img.cols;
                int type = img.type();
                ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
                ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                ofs.write(reinterpret_cast<const char*>(&type), sizeof(type));
                // Write image data
                ofs.write(reinterpret_cast<const char*>(img.data), img.total() * img.elemSize());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing to file: " << e.what() << std::endl;
    }
    ofs.close();
}
cv::Mat cvLib::subclasses::computeDescriptors(std::vector<cv::KeyPoint>& keypoints) {
    // Check for empty keypoints
    if (keypoints.empty()) {
        std::cerr << "No keypoints provided." << std::endl;
        return cv::Mat(); // Handle the error appropriately
    }
    cv::Mat test_descriptors;
    try {
        cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_FEATURES);
        // Validate dummy image size
        if (ReSIZE_IMG_WIDTH <= 0 || ReSIZE_IMG_HEIGHT <= 0) {
            std::cerr << "Invalid dimensions for dummy_image: " 
                      << ReSIZE_IMG_WIDTH << "x" << ReSIZE_IMG_HEIGHT << std::endl;
            return cv::Mat();
        }
        // Create a dummy Mat to store the keypoints
        cv::Mat dummy_image = cv::Mat::zeros(cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT), CV_8UC1);
        // Check validity of keypoints against dummy image
        for (const auto& kp : keypoints) {
            if (kp.pt.x < 0 || kp.pt.x >= dummy_image.cols || kp.pt.y < 0 || kp.pt.y >= dummy_image.rows) {
                std::cerr << "Keypoint out of bounds: " << kp.pt << std::endl;
                return cv::Mat(); // or handle this case more appropriately
            }
        }
        // Set the dummy image to a default value (e.g., 255)
        dummy_image.setTo(cv::Scalar(255));
        // Compute the descriptors
        orb->compute(dummy_image, keypoints, test_descriptors);
        if (test_descriptors.empty()) {
            std::cerr << "Failed to compute descriptors." << std::endl;
        }
    }
    catch (cv::Exception& ex) {
        std::cerr << "OpenCV Exception: " << ex.what() << std::endl;
    }
    catch (std::exception& ex) {
        std::cerr << "Standard Exception: " << ex.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error in cvLib::subclasses::computeDescriptors" << std::endl;
    }
    return test_descriptors;
}
bool cvLib::subclasses::img1_img2_are_matched(const std::string& img1, const std::string& img2){
    if (img1.empty() || img2.empty()) {  
        std::cerr << "Image paths are empty." << std::endl;  
        return false;  
    }  
    cv::Mat m_img1;
    preprocessImg(img1,m_img1);
    cv::Mat m_img2;
    preprocessImg(img2,m_img2);
    if (m_img1.empty() || m_img2.empty()) {  
        std::cerr << "Failed to read one or both images." << std::endl;  
        return false;  
    }  
    // Use ORB for keypoint detection and description
    std::vector<cv::KeyPoint> keypoints1, keypoints2;  
    cv::Mat descriptors1, descriptors2;  
    keypoints1 = extractORBFeatures(m_img1,descriptors1);
    keypoints2 = extractORBFeatures(m_img2,descriptors2);
    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "Existing m_img1 or m_img2 has no descriptors." << std::endl;
        return false;
    }
    /*
     *Start comparing
    */
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    if (goodMatches.size() > DE_THRESHOLD) {
        return true;
    }
    return false;  
}
void cvLib::img_compress(const std::string& input_folder,int quality){
    if(input_folder.empty()){
        return;
    }
    if (!std::filesystem::exists(input_folder)) {
        std::cerr << "The folder does not exist" << std::endl;
        return;
    }
    cvLib::subclasses cvlib_sub;
    try {
        for (const auto& entryMainFolder : std::filesystem::directory_iterator(input_folder)) {  
            if (entryMainFolder.is_directory()) {  
                std::string sub_folder_path = entryMainFolder.path().string();
                std::cout << "Start working on folder: " << sub_folder_path << std::endl;
                for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
                    if (entrySubFolder.is_regular_file()) {   
                        std::string imgFilePath = entrySubFolder.path().string(); 
                        if(imgFilePath.empty()){
                            continue;
                        }
                        if(cvlib_sub.isValidImage(imgFilePath)){
                            try{
                                cvlib_sub.compressJPEG(imgFilePath,imgFilePath,quality);
                            }
                            catch(const std::exception& ex){
                                std::cerr << ex.what() << std::endl;
                            }
                            catch(...){
                                std::cerr << "cvlib_sub.compressJPEG(imgFilePath,imgFilePath,quality); Unknown error!" << std::endl;
                            }
                            std::cout << "Successfully compressed the image: " << imgFilePath << std::endl;
                        }
                    }
                }
            }
        }
    }
    catch(std::exception& ex){
        std::cerr << ex.what() << std::endl;
    }
    catch(...){
        std::cerr << "Unknown errors." << std::endl;
    }
    std::cout << "All jobs are done!" << std::endl;
}
void cvLib::train_img_occurrences(const std::string& images_folder_path, const std::string& output_model_path){
    if(images_folder_path.empty() || output_model_path.empty()){
        return;
    }
    cvLib::subclasses cvlib_sub;//img1_img2_are_matched_return
    std::unordered_map<std::string, std::vector<cv::Mat>> dataset_keypoint;
    try {  
        for (const auto& entryMainFolder : std::filesystem::directory_iterator(images_folder_path)) {  
            if (entryMainFolder.is_directory()) { // Check if the entry is a directory  
                std::string sub_folder_name = entryMainFolder.path().filename().string();  
                std::string sub_folder_path = entryMainFolder.path().string();  
                std::vector<std::vector<cv::KeyPoint>> sub_folder_keypoints;
                std::vector<std::string> sub_folder_file_list;
                std::vector<cv::Mat> sub_folder_imgs;
                for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {  
                    if (entrySubFolder.is_regular_file()) {  
                        std::string imgFilePath = entrySubFolder.path().string(); 
                        if(cvlib_sub.isValidImage(imgFilePath)){
                           sub_folder_file_list.push_back(imgFilePath);
                        }
                    }
                }
                /*
                 * Start processing sub_folder_file_list
                 */
                std::unordered_map<unsigned int,unsigned int> already_saved;
                if(!sub_folder_file_list.empty()){
                    /*Only has one image*/
                    if(sub_folder_file_list.size() == 1){
                        cv::Mat trained_img;
                        cvlib_sub.preprocessImg(sub_folder_file_list[0],trained_img);
                        if(!trained_img.empty()){
                            sub_folder_imgs.push_back(trained_img);
                        }
                        else{
                            std::cerr << "sub_folder_file_list[0] is empty!" << std::endl;
                        }
                    }
                    else{ // more than one image
                        /*initialize already_saved*/
                        for(unsigned int i = 0; i < sub_folder_file_list.size(); ++i){
                            already_saved[i] = 0;
                        }
                        for(unsigned int i = 0; i < sub_folder_file_list.size(); ++i){
                            if(i+1 < sub_folder_file_list.size()){
                                std::cout << "reading: " << sub_folder_file_list[i] << std::endl;
                                if(already_saved[i]==0){
                                    cv::Mat trained_img1;
                                    cvlib_sub.preprocessImg(sub_folder_file_list[i],trained_img1);
                                    sub_folder_imgs.push_back(trained_img1);
                                    already_saved[i] = 1;
                                    for(unsigned int j = 0; j < sub_folder_file_list.size(); ++j){
                                        if(j!=i && already_saved[j]==0){
                                            if(!cvlib_sub.img1_img2_are_matched(sub_folder_file_list[i],sub_folder_file_list[j])){
                                                cv::Mat trained_img2;
                                                cvlib_sub.preprocessImg(sub_folder_file_list[j],trained_img2);
                                                sub_folder_imgs.push_back(trained_img2);
                                                already_saved[j] = 1;
                                            }
                                        }
                                    }
                                }
                            }//if
                        }//for
                    }//else{ // more than one image
                }
                /*
                 * Evaluate images, remove those can not be recognized
                 * */
                if(!sub_folder_imgs.empty()){
                     unsigned int check_num = 0;
                     std::cout << "Evaluating image collections..." << std::endl;
                     for(auto it = sub_folder_imgs.begin(); it != sub_folder_imgs.end();){
                        cv::Mat eval_descriptors;
                        std::vector<cv::KeyPoint> trained_key = cvlib_sub.extractORBFeatures(*it, eval_descriptors);
                        if (eval_descriptors.empty()) {
                            std::cerr << "Removing bad image at index: " << std::to_string(check_num) << std::endl;
                            it = sub_folder_imgs.erase(it);
                        }
                        else{
                            ++it;
                        }
                        check_num++;
                     }
                }
                else{
                    std::cerr << "cvLib::train_img_occurrences sub_folder_imgs is empty!" << std::endl;
                    continue;
                }
                dataset_keypoint[sub_folder_name] = sub_folder_imgs;
                std::cout << sub_folder_name << " training is done!" << std::endl;
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {  
        std::cerr << "Filesystem error: " << e.what() << std::endl;  
        return; // Return an empty dataset in case of filesystem error  
    }  
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }
    if(!dataset_keypoint.empty()){
        cvlib_sub.saveModel_keypoint(dataset_keypoint,output_model_path);
        std::cout << "Successfully saved the model file to " << output_model_path << std::endl;
    }
    else{
        std::cerr << "cvLib::train_img_occurrences dataset_keypoint is empty!" << std::endl;
    }
}
void cvLib::loadModel_keypoint(std::unordered_map<std::string, std::vector<cv::Mat>>& featureMap, const std::string& filename){
    if (filename.empty()) {
        return;
    }
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
        return;
    }
    try {
        size_t mapSize;
        ifs.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));
        for (size_t i = 0; i < mapSize; ++i) {
            size_t keySize;
            ifs.read(reinterpret_cast<char*>(&keySize), sizeof(keySize));
            std::string className(keySize, '\0');
            ifs.read(&className[0], keySize);
            size_t imageCount;
            ifs.read(reinterpret_cast<char*>(&imageCount), sizeof(imageCount));
            std::vector<cv::Mat> images(imageCount);
            for (size_t j = 0; j < imageCount; ++j) {
                // Read image dimensions
                int rows, cols, type;
                ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
                ifs.read(reinterpret_cast<char*>(&type), sizeof(type));
                // Create an empty Mat to hold the image
                cv::Mat img(rows, cols, type);
                // Read image data
                ifs.read(reinterpret_cast<char*>(img.data), img.total() * img.elemSize());
                images[j] = img; // Store the image in the vector
            }
            featureMap[className] = images; // Store the vector of images in the map
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading from file: " << e.what() << std::endl;
    }
    ifs.close();
}
void cvLib::ini_trained_data(const std::string& model_path){
    if(model_path.empty()){
        std::cerr << "Model file path is empty!" << std::endl;
        return;
    }
    loadModel_keypoint(trained_dataset, model_path);
}
void cvLib::img_recognition(const std::vector<std::string>& input_images_path, std::unordered_map<std::string, return_img_info>& return_imgs) {
    if (input_images_path.empty()) {
        std::cerr << "Error: Model file path or input images path is empty." << std::endl;
        return;
    }
    if (trained_dataset.empty()) {
        std::cerr << "cvLib::img_recognition: trained_dataset is empty!" << std::endl;
        return;
    }
    cvLib::subclasses cvlib_sub;
    for (const auto& test_item : input_images_path) {
        std::vector<std::pair<std::string, unsigned int>> score_count;
        auto t_count_start = std::chrono::high_resolution_clock::now();
        return_img_info rii;
        cv::Mat test_img;
        cvlib_sub.preprocessImg(test_item, test_img);
        if (test_img.empty()) {
            std::cerr << "cvLib::img_recognition: preprocessImg, output test_img is empty!" << std::endl;
            continue; // Skip empty images
        }
        cv::Mat test_descriptors;
        std::vector<cv::KeyPoint> sub_key = cvlib_sub.extractORBFeatures(test_img, test_descriptors);
        if (test_descriptors.empty()) {
            std::cerr << "cvLib::img_recognition: test_img->descriptors is empty!" << std::endl;
            continue; // Skip if no descriptors are found
        }
        for (const auto& item : trained_dataset) { // Process trained corpus
            auto item_collections = item.second;
            for (const auto& trained_item : item_collections) {
                cv::Mat trained_descriptors;
                std::vector<cv::KeyPoint> trained_key = cvlib_sub.extractORBFeatures(trained_item, trained_descriptors);
                if (trained_descriptors.empty()) {
                    std::cerr << "Warning: trained descriptors are empty for trained_item." << std::endl;
                    continue; // Skip this trained item if no descriptors found
                }
                cv::BFMatcher matcher(cv::NORM_L2);
                std::vector<std::vector<cv::DMatch>> knnMatches;
                matcher.knnMatch(test_descriptors, trained_descriptors, knnMatches, 2);
                std::vector<cv::DMatch> goodMatches;
                for (const auto& match : knnMatches) {
                    if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                        goodMatches.push_back(match[0]);
                    }
                }
                if (goodMatches.size() > DE_THRESHOLD) {//DE_THRESHOLD
                    //std::cout << test_item << " trained item: " << item.first << " score: " << goodMatches.size() << std::endl;
                    score_count.push_back(std::make_pair(item.first,goodMatches.size()));
                }
                //std::cout << test_item << " trained item: " << item.first << " score: " << goodMatches.size() << std::endl;
            }
        }
        if(!score_count.empty()){
            std::unordered_map<std::string, unsigned int> max_scores;
            // Iterate through the score_count vector
            for (const auto& [key, value] : score_count) {
                // Update the maximum score for the current key
                max_scores[key] = std::max(max_scores[key], value);
            }
            std::vector<std::pair<std::string, double>> sorted_score_counting(max_scores.begin(), max_scores.end());
            // Sort the vector of pairs
            std::sort(sorted_score_counting.begin(), sorted_score_counting.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            auto it = sorted_score_counting.begin();
            rii.objName = it->first;
             auto t_count_end = std::chrono::high_resolution_clock::now();
            rii.timespent = t_count_end - t_count_start;
            return_imgs[test_item] = rii; // Store results
        }
    }
}
void cvLib::checkExistingGestures(const cv::Mat& frame_input, std::string& gesture_catched){
    if(frame_input.empty()){
        std::cerr << "cvLib::checkExistingGestures frame input is empty!" << std::endl;
        return;
    }
    if(trained_dataset.empty()){
        std::cerr << "cvLib::checkExistingGestures trained_dataset is empty!" << std::endl;
        return;
    }
    cv::Mat test_img;
    std::vector<std::pair<std::string, unsigned int>> score_count;
    try {
        cv::Mat resizeImg;
        cv::resize(frame_input, resizeImg, cv::Size(ReSIZE_IMG_WIDTH, ReSIZE_IMG_HEIGHT));
        cv::Mat img_gray;
        cv::cvtColor(resizeImg, img_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(img_gray, test_img, cv::Size(5, 5), 0);
    } catch(const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return;
    } catch(...) {
        std::cerr << "cvLib::checkExistingGestures try preprocess image unknown errors" << std::endl;
        return;
    }
    if (test_img.empty()) {
        std::cerr << "cvLib::img_recognition: preprocessImg, output test_img is empty!" << std::endl;
        return; // Skip empty images
    }
    cvLib::subclasses cvlib_sub;
    cv::Mat test_descriptors;
    std::vector<cv::KeyPoint> sub_key = cvlib_sub.extractORBFeatures(test_img, test_descriptors);
    if (test_descriptors.empty()) {
        std::cerr << "cvLib::img_recognition: test_img->descriptors is empty!" << std::endl;
        return; // Skip if no descriptors are found
    }
    for (const auto& item : trained_dataset) { // Process trained corpus
        auto item_collections = item.second;
        for (const auto& trained_item : item_collections) {
            cv::Mat trained_descriptors;
            std::vector<cv::KeyPoint> trained_key = cvlib_sub.extractORBFeatures(trained_item, trained_descriptors);
            if (trained_descriptors.empty()) {
                std::cerr << "Warning: trained descriptors are empty for trained_item." << std::endl;
                continue; // Skip this trained item if no descriptors found
            }
            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher.knnMatch(test_descriptors, trained_descriptors, knnMatches, 2);
            std::vector<cv::DMatch> goodMatches;
            for (const auto& match : knnMatches) {
                if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                    goodMatches.push_back(match[0]);
                }
            }
            if (goodMatches.size() > DE_THRESHOLD) { // DE_THRESHOLD
                score_count.push_back(std::make_pair(item.first, goodMatches.size()));
                // Assuming you have the coordinates of the recognized object
                // For demonstration, let's use dummy coordinates for the rectangle
                cv::Point topLeft(50, 50);    // Top-left corner of the rectangle
                cv::Point bottomRight(150, 150); // Bottom-right corner of the rectangle
                cv::rectangle(frame_input, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2); // Draw rectangle
            }
        }
    }
    if (!score_count.empty()) {
        std::unordered_map<std::string, unsigned int> max_scores;
        // Iterate through the score_count vector
        for (const auto& [key, value] : score_count) {
            // Update the maximum score for the current key
            max_scores[key] = std::max(max_scores[key], value);
        }
        std::vector<std::pair<std::string, double>> sorted_score_counting(max_scores.begin(), max_scores.end());
        // Sort the vector of pairs
        std::sort(sorted_score_counting.begin(), sorted_score_counting.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        auto it = sorted_score_counting.begin();
        gesture_catched = it->first;
    }
}
bool cvLib::checkExistingFace(const std::string& faces_folder_path, const cv::Mat& img_input) {
    if (!std::filesystem::exists(faces_folder_path) || !std::filesystem::is_directory(faces_folder_path)) {
        std::cerr << "The folder does not exist or cannot be accessed." << std::endl;
        return false;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_input, img_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(img_gray, img_gray, cv::Size(5, 5), 0);
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(MAX_FEATURES);
    std::vector<cv::KeyPoint> keypoints_input;
    cv::Mat descriptors_input;
    detector->detectAndCompute(img_gray, cv::noArray(), keypoints_input, descriptors_input);
    if (descriptors_input.empty()) {
        std::cerr << "Input image has no descriptors." << std::endl;
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(faces_folder_path)) {
        if (entry.is_regular_file()) {
            cv::Mat existing_face = cv::imread(entry.path().string());
            if (existing_face.empty()) continue;
            cv::Mat existing_face_gray;
            cv::cvtColor(existing_face, existing_face_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints_existing;
            cv::Mat descriptors_existing;
            detector->detectAndCompute(existing_face_gray, cv::noArray(), keypoints_existing, descriptors_existing);
            if (descriptors_existing.empty()) {
                std::cerr << "Existing face has no descriptors." << std::endl;
                continue;
            }
            cv::BFMatcher matcher(cv::NORM_L2);
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher.knnMatch(descriptors_existing, descriptors_input, knnMatches, 2);
            std::vector<cv::DMatch> goodMatches;
            for (const auto& match : knnMatches) {
                if (match.size() > 1 && match[0].distance < RATIO_THRESH * match[1].distance) {
                    goodMatches.push_back(match[0]);
                }
            }
            if (goodMatches.size() > DE_THRESHOLD) {
                //std::cout << "The face image already exists." << std::endl;
                return true;
            }
        }
    }
    return false;
}
void cvLib::onFacesDetected(const std::vector<cv::Rect>& faces, cv::Mat& frame, const std::string& face_folder) {
    if (faces.empty()) return;
    for (size_t i = 0; i < faces.size(); ++i) {
        cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2);
        std::string text = (i < 9) ? "0" + std::to_string(i + 1) : std::to_string(i + 1);
        cv::putText(frame, text, cv::Point(faces[i].x, faces[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }
    cv::Rect faceROI = faces[0]; // Save the first detected face
    cv::Mat faceImage = frame(faceROI).clone();
    if (checkExistingFace(face_folder, faceImage)) {
        return;
    }
    std::string fileName = face_folder + "/face_" + std::to_string(faceCount++) + ".jpg";
    cv::imwrite(fileName, faceImage);
    std::cout << "Saved snapshot: " << fileName << std::endl;
    /*
     Detect gestures 
     */
    
}
void cvLib::start_recording(unsigned int webcamIndex){
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("/Users/dengfengji/ronnieji/MLCpplib-main/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load Haar Cascade model." << std::endl;
        return;
    }
    cv::VideoCapture cap(webcamIndex);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream." << std::endl;
        return;
    }
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame." << std::endl;
            break;
        }
        cv::Mat gestrueFrame = frame.clone();
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 10, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        if (!faces.empty()) {
            onFacesDetected(faces, frame, "/Users/dengfengji/ronnieji/MLCpplib-main/faces");
        }
        std::string catched_gesture;
        checkExistingGestures(gestrueFrame,catched_gesture);
        if(!catched_gesture.empty()){
            std::cout << "Gesture catched: " << catched_gesture << std::endl;
        }
        cv::imshow("Face Detection", frame);
        char key = cv::waitKey(30);
        if (key == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}