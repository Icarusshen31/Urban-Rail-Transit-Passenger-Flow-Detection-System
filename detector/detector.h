#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct DetectorConfig {
    int INPUT_WIDTH = 320;
    int INPUT_HEIGHT = 240;
    float CONFIDENCE_THRESHOLD = 0.5f;
    float NMS_THRESHOLD = 0.4f;
};

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;

    Detection() : bbox(0, 0, 0, 0), confidence(0.0f), class_id(0) {}
    Detection(const cv::Rect2f& box, float conf, int cls)
        : bbox(box), confidence(conf), class_id(cls) {}

    cv::Point2f center() const {
        return cv::Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
    }

    float area() const { return bbox.width * bbox.height; }

    std::vector<float> toArray() const {
        return { bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height,
                confidence, static_cast<float>(class_id) };
    }
};

class YOLODetector {
public:
    YOLODetector(const DetectorConfig& config = DetectorConfig(),
        const std::string& model_path = "");
    ~YOLODetector();

    std::vector<Detection> detect(const cv::Mat& image);

private:
    DetectorConfig config_;
    std::string model_path_;
    cv::dnn::Net net_;
    bool use_rknn_ = false;  // placeholder, not implemented
    void* rknn_handle_ = nullptr;

    void loadModel();
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Detection> postprocess(const cv::Mat& outputs,
        const cv::Size& original_size);
    std::vector<Detection> nms(const std::vector<Detection>& detections);
    std::vector<Detection> simulateDetection(const cv::Mat& image);
};

class DummyDetector {
public:
    DummyDetector() = default;
    std::vector<Detection> detect(const cv::Mat&) { return {}; }
};

// Utility functions
cv::Mat drawDetections(const cv::Mat& image,
    const std::vector<Detection>& detections,
    const cv::Scalar& color = cv::Scalar(0, 255, 0),
    int thickness = 2);

std::vector<float> detectionsToArray(const std::vector<Detection>& detections);

#endif // DETECTOR_HPP