#include "detector.hpp"
#include <iostream>
#include <algorithm>

YOLODetector::YOLODetector(const DetectorConfig& config, const std::string& model_path)
    : config_(config), model_path_(model_path) {
    loadModel();
}

YOLODetector::~YOLODetector() {
    // if rknn_handle_ not null, release (not implemented)
}

void YOLODetector::loadModel() {
    if (model_path_.empty()) {
        std::cout << "Warning: no model path, using simulation mode" << std::endl;
        return;
    }
    try {
        if (model_path_.find(".onnx") != std::string::npos) {
            net_ = cv::dnn::readNetFromONNX(model_path_);
            std::cout << "Loaded ONNX model: " << model_path_ << std::endl;
        }
        else if (model_path_.find(".rknn") != std::string::npos) {
            std::cout << "RKNN not supported in this C++ version, falling back to simulation" << std::endl;
            // Placeholder for RKNN, we skip
        }
        else {
            std::cout << "Unsupported model format: " << model_path_ << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "Model load failed: " << e.what() << ", using simulation mode" << std::endl;
        net_.clear();
    }
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& image) {
    if (net_.empty()) {
        return simulateDetection(image);
    }

    cv::Mat blob = preprocess(image);
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    std::vector<Detection> detections = postprocess(outputs, image.size());
    return nms(detections);
}

cv::Mat YOLODetector::preprocess(const cv::Mat& image) {
    cv::Mat input = image;
    if (input.channels() == 1) {
        cv::cvtColor(input, input, cv::COLOR_GRAY2BGR);
    }

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(config_.INPUT_WIDTH, config_.INPUT_HEIGHT));

    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0 / 255.0,
        cv::Size(config_.INPUT_WIDTH, config_.INPUT_HEIGHT),
        cv::Scalar(), false, false);
    return blob;
}

std::vector<Detection> YOLODetector::postprocess(const cv::Mat& outputs,
    const cv::Size& original_size) {
    std::vector<Detection> detections;

    // Assume output shape: [1, num_boxes, 5+num_classes] or [num_boxes, 5+num_classes]
    cv::Mat out = outputs;
    if (out.dims == 3) {
        out = out.reshape(1, out.size[1]); // flatten first two dims -> rows
    }

    float scale_x = static_cast<float>(original_size.width) / config_.INPUT_WIDTH;
    float scale_y = static_cast<float>(original_size.height) / config_.INPUT_HEIGHT;

    for (int i = 0; i < out.rows; ++i) {
        float* row = out.ptr<float>(i);
        float x1 = row[0] * scale_x;
        float y1 = row[1] * scale_y;
        float x2 = row[2] * scale_x;
        float y2 = row[3] * scale_y;
        float conf = row[4];

        if (conf < config_.CONFIDENCE_THRESHOLD) continue;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_size.width - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_size.height - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(original_size.width - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(original_size.height - 1)));

        int class_id = 0;
        if (out.cols > 5) {
            float* class_scores = row + 5;
            int num_classes = out.cols - 5;
            int max_idx = std::max_element(class_scores, class_scores + num_classes) - class_scores;
            class_id = max_idx;
        }

        cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);
        detections.emplace_back(box, conf, class_id);
    }

    return detections;
}

std::vector<Detection> YOLODetector::nms(const std::vector<Detection>& detections) {
    if (detections.empty()) return {};

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    for (const auto& d : detections) {
        boxes.push_back(d.bbox);
        scores.push_back(d.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, config_.CONFIDENCE_THRESHOLD, config_.NMS_THRESHOLD, indices);

    std::vector<Detection> result;
    for (int idx : indices) {
        result.push_back(detections[idx]);
    }
    return result;
}

std::vector<Detection> YOLODetector::simulateDetection(const cv::Mat& image) {
    std::vector<Detection> detections;

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::Mat binary;
    cv::threshold(gray, binary, 50, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        double area = cv::contourArea(contour);
        if (area < 200 || rect.width < 10 || rect.height < 15) continue;

        cv::Rect2f box(rect.x, rect.y, rect.width, rect.height);
        detections.emplace_back(box, 0.9f, 0);
    }

    return detections;
}

cv::Mat drawDetections(const cv::Mat& image,
    const std::vector<Detection>& detections,
    const cv::Scalar& color,
    int thickness) {
    cv::Mat result = image.clone();
    for (const auto& d : detections) {
        cv::rectangle(result, d.bbox, color, thickness);
        std::string label = cv::format("%.2f", d.confidence);
        cv::putText(result, label,
            cv::Point(static_cast<int>(d.bbox.x),
                static_cast<int>(d.bbox.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    return result;
}

std::vector<float> detectionsToArray(const std::vector<Detection>& detections) {
    std::vector<float> arr;
    arr.reserve(detections.size() * 6);
    for (const auto& d : detections) {
        auto v = d.toArray();
        arr.insert(arr.end(), v.begin(), v.end());
    }
    return arr;
}