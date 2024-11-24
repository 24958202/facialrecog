/*
    Program to detect faces in a video and save unique face images.
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
const unsigned int MAX_FEATURES = 1000;   // Max number of features to detect
const float RATIO_THRESH = 0.95f;          // Ratio threshold for matching
const unsigned int DE_THRESHOLD = 10;      // Min matches to consider a face as existing
unsigned int faceCount = 0;                 // Counter for faces detected
bool checkExistingFace(const std::string& faces_folder_path, const cv::Mat& img_input) {
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
                std::cout << "The face image already exists." << std::endl;
                return true;
            }
        }
    }
    return false;
}
void onFacesDetected(const std::vector<cv::Rect>& faces, cv::Mat& frame, const std::string& face_folder) {
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
}
void startRecording() {
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load("/Users/dengfengji/ronnieji/MLCpplib-main/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error: Could not load Haar Cascade model." << std::endl;
        return;
    }
    cv::VideoCapture cap(0);
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
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 10, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
        if (!faces.empty()) {
            onFacesDetected(faces, frame, "/Users/dengfengji/ronnieji/MLCpplib-main/faces");
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
int main() {
    startRecording();
    return 0;
}
