#ifndef DOOR_DISTANCE_H
#define DOOR_DISTANCE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <deque>
#include <string>
#include <utility>

// 车门距离计算配置结构体
struct DoorDistanceConfig {
    // ROI区域比例：左、上、右、下
    float ROI_RATIO[4] = { 0.1f, 0.2f, 0.9f, 0.8f };
    // 梯度计算窗口大小
    int WINDOW_SIZE = 5;
    // 边界梯度阈值
    float GRADIENT_THRESHOLD = 50.0f;
    // 深度相机内参
    float CX = 160.0f;
    float CY = 120.0f;
    float FX = 300.0f;
    float FY = 300.0f;
    // 距离平滑帧数
    int SMOOTH_FRAMES = 5;
};

// 车门边界结果结构体
struct DoorBoundary {
    cv::Point3f left_point;   // 左边界3D坐标
    cv::Point3f right_point;  // 右边界3D坐标
    float distance;           // 车门宽度（mm）
    float confidence;         // 检测置信度
};

// 车门距离计算器类
class DoorDistanceCalculator {
private:
    DoorDistanceConfig config;
    int image_height;
    int image_width;
    int roi_x1, roi_y1, roi_x2, roi_y2;
    std::deque<float> distance_history;  // 历史距离（平滑用）

    // 计算深度梯度
    float calculateGradient(const cv::Mat& depth, int x, int y, const std::string& dir);
    // 查找车门左右边界点
    std::pair<cv::Point3i, cv::Point3i> findBoundaryPoints(const cv::Mat& roi_depth);
    // 像素坐标转3D坐标
    cv::Point3f pixelTo3d(const cv::Point3i& pt);
    // 3D点欧氏距离
    float euclideanDistance(const cv::Point3f& p1, const cv::Point3f& p2);
    // 计算检测置信度
    float calculateConfidence(const cv::Mat& roi_depth, const cv::Point3i& left, const cv::Point3i& right);

public:
    // 构造函数
    DoorDistanceCalculator(const DoorDistanceConfig& cfg, cv::Size img_shape);
    // 提取ROI区域
    cv::Mat extractRoi(const cv::Mat& depth_image);
    // 核心计算：获取车门边界和宽度
    DoorBoundary calculate(const cv::Mat& depth_image);
    // 判断车门是否打开
    bool isDoorOpen(float threshold_mm);
    // 获取车门状态字符串
    std::string getDoorStatus();
};

// 绘制车门边界可视化
cv::Mat drawDoorBoundary(const cv::Mat& image, const DoorBoundary& boundary, cv::Scalar color = cv::Scalar(0, 255, 0));

#endif // DOOR_DISTANCE_H