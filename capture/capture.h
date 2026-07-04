#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct PreprocessConfig {
    int OUTPUT_WIDTH = 160;
    int OUTPUT_HEIGHT = 120;
    int MORPH_KERNEL_SIZE = 3;
    int NOISE_AREA_THRESHOLD = 30;
};

class DepthImagePreprocessor {
public:
    DepthImagePreprocessor(const PreprocessConfig& config = PreprocessConfig());
    bool preprocess(const cv::Mat& depth_image, cv::Mat& processed, cv::Mat& mask);

private:
    PreprocessConfig config_;
    cv::Mat morph_kernel_;

    cv::Mat resize(const cv::Mat& image);
    cv::Mat otsuSegmentation(const cv::Mat& depth_image);
    cv::Mat morphologyDenoise(const cv::Mat& mask);
    cv::Mat applyMaskAndNormalize(const cv::Mat& depth_image, const cv::Mat& mask);
};

class TOFCameraCapture {
public:
    TOFCameraCapture(const std::string& source_type = "simulator",
        const std::string& source_path = "");
    ~TOFCameraCapture();

    bool read(cv::Mat& depth);
    void release();

private:
    std::string source_type_;
    std::string source_path_;
    cv::VideoCapture camera_;
    cv::VideoCapture video_capture_;
    std::vector<std::string> image_list_;
    size_t current_index_ = 0;

    void initSource();
    bool readSimulated(cv::Mat& depth);
    bool readCamera(cv::Mat& depth);
    bool readVideo(cv::Mat& depth);
    bool readImage(cv::Mat& depth);
};

#endif // CAPTURE_HPP