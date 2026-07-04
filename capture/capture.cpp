#include "capture.hpp"
#include <random>
#include <filesystem>
#include <algorithm>

DepthImagePreprocessor::DepthImagePreprocessor(const PreprocessConfig& config)
    : config_(config) {
    morph_kernel_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(config_.MORPH_KERNEL_SIZE, config_.MORPH_KERNEL_SIZE));
}

bool DepthImagePreprocessor::preprocess(const cv::Mat& depth_image,
                                        cv::Mat& processed,
                                        cv::Mat& mask) {
    if (depth_image.empty()) return false;
    cv::Mat img = depth_image;
    if (img.channels() > 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    if (img.type() != CV_16UC1) {
        img.convertTo(img, CV_16UC1);
    }

    cv::Mat resized = resize(img);
    mask = otsuSegmentation(resized);
    mask = morphologyDenoise(mask);
    processed = applyMaskAndNormalize(resized, mask);
    return true;
}

cv::Mat DepthImagePreprocessor::resize(const cv::Mat& image) {
    cv::Mat dst;
    cv::resize(image, dst,
               cv::Size(config_.OUTPUT_WIDTH, config_.OUTPUT_HEIGHT),
               0, 0, cv::INTER_NEAREST);
    return dst;
}

cv::Mat DepthImagePreprocessor::otsuSegmentation(const cv::Mat& depth_image) {
    cv::Mat valid = depth_image.clone();
    valid.setTo(65535, valid == 0);

    cv::Mat depth_8bit;
    cv::normalize(valid, depth_8bit, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat mask;
    cv::threshold(depth_8bit, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    return mask;
}

cv::Mat DepthImagePreprocessor::morphologyDenoise(const cv::Mat& mask) {
    cv::Mat eroded, dilated;
    cv::erode(mask, eroded, morph_kernel_, cv::Point(-1,-1), 1);
    cv::dilate(eroded, dilated, morph_kernel_, cv::Point(-1,-1), 1);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(dilated, labels, stats, centroids, 8);

    cv::Mat cleaned = cv::Mat::zeros(mask.size(), mask.type());
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > config_.NOISE_AREA_THRESHOLD) {
            cleaned.setTo(255, labels == i);
        }
    }
    return cleaned;
}

cv::Mat DepthImagePreprocessor::applyMaskAndNormalize(const cv::Mat& depth_image,
                                                      const cv::Mat& mask) {
    cv::Mat masked = depth_image.clone();
    masked.setTo(0, mask == 0);

    cv::Mat output = cv::Mat::zeros(masked.size(), CV_8U);
    cv::Mat nonzero_mask;
    cv::compare(masked, 0, nonzero_mask, cv::CMP_GT);

    if (cv::countNonZero(nonzero_mask) > 0) {
        double min_val, max_val;
        cv::minMaxLoc(masked, &min_val, &max_val, nullptr, nullptr, nonzero_mask);
        if (max_val > min_val) {
            cv::Mat normalized;
            masked.convertTo(normalized, CV_32F);
            normalized = (normalized - min_val) / (max_val - min_val) * 255.0;
            normalized.convertTo(output, CV_8U);
            output.setTo(0, mask == 0);
        }
    }
    return output;
}

TOFCameraCapture::TOFCameraCapture(const std::string& source_type,
                                   const std::string& source_path)
    : source_type_(source_type), source_path_(source_path) {
    initSource();
}

TOFCameraCapture::~TOFCameraCapture() {
    release();
}

void TOFCameraCapture::initSource() {
    if (source_type_ == "camera") {
        camera_.open(0);
        if (!camera_.isOpened()) {
            source_type_ = "simulator";
        }
    } else if (source_type_ == "video") {
        video_capture_.open(source_path_);
    } else if (source_type_ == "image_folder") {
        namespace fs = std::filesystem;
        for (const auto& entry : fs::directory_iterator(source_path_)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                    image_list_.push_back(entry.path().string());
                }
            }
        }
        std::sort(image_list_.begin(), image_list_.end());
    }
}

bool TOFCameraCapture::read(cv::Mat& depth) {
    if (source_type_ == "simulator") return readSimulated(depth);
    if (source_type_ == "camera")   return readCamera(depth);
    if (source_type_ == "video")    return readVideo(depth);
    if (source_type_ == "image_folder") return readImage(depth);
    return false;
}

bool TOFCameraCapture::readSimulated(cv::Mat& depth) {
    int h = PreprocessConfig().OUTPUT_HEIGHT;
    int w = PreprocessConfig().OUTPUT_WIDTH;
    depth = cv::Mat(h, w, CV_16UC1, cv::Scalar(3000));

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> num_people(1, 4);
    int n = num_people(gen);
    for (int i = 0; i < n; ++i) {
        std::uniform_int_distribution<> cx(50, w-50);
        std::uniform_int_distribution<> cy(50, h-50);
        std::uniform_int_distribution<> rx(15, 30);
        std::uniform_int_distribution<> ry(20, 40);
        int cx_ = cx(gen), cy_ = cy(gen);
        int rx_ = rx(gen), ry_ = ry(gen);
        int depth_val = 1000 + std::uniform_int_distribution<>(0, 500)(gen);
        cv::ellipse(depth, cv::Point(cx_, cy_),
                    cv::Size(rx_, ry_), 0, 0, 360,
                    cv::Scalar(depth_val), -1);
    }

    cv::Mat noise(h, w, CV_16UC1);
    cv::randu(noise, cv::Scalar(0), cv::Scalar(100));
    cv::add(depth, noise, depth);
    return true;
}

bool TOFCameraCapture::readCamera(cv::Mat& depth) {
    if (!camera_.isOpened()) return false;
    cv::Mat frame;
    if (!camera_.read(frame)) return false;
    if (frame.channels() > 1) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }
    frame.convertTo(depth, CV_16UC1);
    return true;
}

bool TOFCameraCapture::readVideo(cv::Mat& depth) {
    if (!video_capture_.isOpened()) return false;
    cv::Mat frame;
    if (!video_capture_.read(frame)) return false;
    if (frame.channels() > 1) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frame.convertTo(depth, CV_16UC1);
        depth *= 10;
    } else {
        frame.convertTo(depth, CV_16UC1);
    }
    return true;
}

bool TOFCameraCapture::readImage(cv::Mat& depth) {
    if (current_index_ >= image_list_.size()) {
        current_index_ = 0;
        return false;
    }
    cv::Mat img = cv::imread(image_list_[current_index_++], cv::IMREAD_UNCHANGED);
    if (img.empty()) return false;
    if (img.channels() > 1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    img.convertTo(depth, CV_16UC1);
    return true;
}

void TOFCameraCapture::release() {
    if (camera_.isOpened()) camera_.release();
    if (video_capture_.isOpened()) video_capture_.release();
}