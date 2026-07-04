#include "door_distance.h"
#include <numeric>
#include <cmath>
#include <cstdio>

DoorDistanceCalculator::DoorDistanceCalculator(const DoorDistanceConfig& cfg, cv::Size img_shape) {
    config = cfg;
    image_height = img_shape.height;
    image_width = img_shape.width;

    roi_x1 = static_cast<int>(image_width * config.ROI_RATIO[0]);
    roi_y1 = static_cast<int>(image_height * config.ROI_RATIO[1]);
    roi_x2 = static_cast<int>(image_width * config.ROI_RATIO[2]);
    roi_y2 = static_cast<int>(image_height * config.ROI_RATIO[3]);
}

cv::Mat DoorDistanceCalculator::extractRoi(const cv::Mat& depth_image) {
    if (depth_image.empty() || roi_x2 <= roi_x1 || roi_y2 <= roi_y1)
        return cv::Mat();
    return depth_image(cv::Rect(roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)).clone();
}

float DoorDistanceCalculator::calculateGradient(const cv::Mat& depth, int x, int y, const std::string& dir) {
    int win = config.WINDOW_SIZE;
    int half = win / 2;
    if (x - win < 0 || x + win >= depth.cols) return 0.f;

    cv::Mat left_reg, right_reg;
    if (dir == "left") {
        left_reg = depth(cv::Range(y - half, y + half + 1), cv::Range(x - win, x));
        right_reg = depth(cv::Range(y - half, y + half + 1), cv::Range(x, x + win));
    }
    else {
        left_reg = depth(cv::Range(y - half, y + half + 1), cv::Range(x, x + win));
        right_reg = depth(cv::Range(y - half, y + half + 1), cv::Range(x - win, x));
    }
    cv::Scalar m1 = cv::mean(left_reg);
    cv::Scalar m2 = cv::mean(right_reg);
    return fabs(m2[0] - m1[0]);
}

std::pair<cv::Point3i, cv::Point3i> DoorDistanceCalculator::findBoundaryPoints(const cv::Mat& roi_depth) {
    int h = roi_depth.rows;
    int w = roi_depth.cols;
    int win = config.WINDOW_SIZE;
    cv::Point3i lp(-1, -1, -1), rp(-1, -1, -1);
    float max_lg = 0, max_rg = 0;

    for (int y = win; y < h - win; y++) {
        for (int x = win; x < w / 2; x++) {
            float g = calculateGradient(roi_depth, x, y, "left");
            if (g > max_lg && g > config.GRADIENT_THRESHOLD) {
                max_lg = g;
                lp = cv::Point3i(x + roi_x1, y + roi_y1, roi_depth.at<ushort>(y, x));
            }
        }
        for (int x = w - win; x > w / 2; x--) {
            float g = calculateGradient(roi_depth, x, y, "right");
            if (g > max_rg && g > config.GRADIENT_THRESHOLD) {
                max_rg = g;
                rp = cv::Point3i(x + roi_x1, y + roi_y1, roi_depth.at<ushort>(y, x));
            }
        }
    }
    return { lp, rp };
}

cv::Point3f DoorDistanceCalculator::pixelTo3d(const cv::Point3i& pt) {
    float u = pt.x, v = pt.y, Z = (float)pt.z;
    float X = (u - config.CX) * Z / config.FX;
    float Y = (v - config.CY) * Z / config.FY;
    return cv::Point3f(X, Y, Z);
}

float DoorDistanceCalculator::euclideanDistance(const cv::Point3f& p1, const cv::Point3f& p2) {
    float dx = p1.x - p2.x, dy = p1.y - p2.y, dz = p1.z - p2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

float DoorDistanceCalculator::calculateConfidence(const cv::Mat& roi_depth, const cv::Point3i& left, const cv::Point3i& right) {
    if (left.x == -1 || right.x == -1) return 0.f;
    float conf = 1.f;
    if (left.z == 0 || right.z == 0) conf *= 0.5f;
    int px_dist = abs(left.x - right.x);
    if (px_dist < 50) conf *= 0.5f;
    else if (px_dist > 300) conf *= 0.7f;
    return conf;
}

DoorBoundary DoorDistanceCalculator::calculate(const cv::Mat& depth_image) {
    DoorBoundary res{ 0 };
    cv::Mat roi = extractRoi(depth_image);
    if (roi.empty()) return res;
    auto [lp, rp] = findBoundaryPoints(roi);
    if (lp.x == -1 || rp.x == -1) return res;

    cv::Point3f l3d = pixelTo3d(lp);
    cv::Point3f r3d = pixelTo3d(rp);
    float dist = euclideanDistance(l3d, r3d);

    distance_history.push_back(dist);
    if (distance_history.size() > config.SMOOTH_FRAMES) distance_history.pop_front();
    float avg = std::accumulate(distance_history.begin(), distance_history.end(), 0.f) / distance_history.size();

    res.left_point = l3d;
    res.right_point = r3d;
    res.distance = avg;
    res.confidence = calculateConfidence(roi, lp, rp);
    return res;
}

bool DoorDistanceCalculator::isDoorOpen(float threshold_mm) {
    if (distance_history.empty()) return false;
    float avg = std::accumulate(distance_history.begin(), distance_history.end(), 0.f) / distance_history.size();
    return avg > threshold_mm;
}

std::string DoorDistanceCalculator::getDoorStatus() {
    if (distance_history.empty()) return "未知";
    float avg = std::accumulate(distance_history.begin(), distance_history.end(), 0.f) / distance_history.size();
    if (avg < 300) return "关闭";
    else if (avg < 600) return "部分打开";
    return "完全打开";
}

cv::Mat drawDoorBoundary(const cv::Mat& image, const DoorBoundary& boundary, cv::Scalar color) {
    cv::Mat out = image.clone();
    if (boundary.confidence <= 0) return out;
    int lx = (int)boundary.left_point.x, ly = (int)boundary.left_point.y;
    int rx = (int)boundary.right_point.x, ry = (int)boundary.right_point.y;
    cv::circle(out, cv::Point(lx, ly), 5, color, -1);
    cv::circle(out, cv::Point(rx, ry), 5, color, -1);
    cv::line(out, cv::Point(lx, ly), cv::Point(rx, ry), color, 2);
    float m = boundary.distance / 1000.f;
    std::string text = cv::format("Door: %.2fm", m);
    cv::putText(out, text, cv::Point(lx, ly - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    return out;
}

// 测试主函数
int main() {
    DoorDistanceCalculator calc(DoorDistanceConfig(), cv::Size(320, 240));
    cv::Mat depth = cv::Mat::ones(240, 320, CV_16U) * 2000;
    for (int y = 80; y < 160; y++) {
        for (int x = 50; x < 150; x++) depth.at<ushort>(y, x) = 1000;
        for (int x = 170; x < 270; x++) depth.at<ushort>(y, x) = 1000;
    }
    DoorBoundary ret = calc.calculate(depth);
    if (ret.confidence > 0) {
        printf("车门宽度: %.2f m\n", ret.distance / 1000.f);
        printf("左(%.2f,%.2f,%.2f) 右(%.2f,%.2f,%.2f)\n",
            ret.left_point.x, ret.left_point.y, ret.left_point.z,
            ret.right_point.x, ret.right_point.y, ret.right_point.z);
        printf("置信度: %.2f 状态:%s\n", ret.confidence, calc.getDoorStatus().c_str());
    }
    else {
        printf("未检测到车门\n");
    }
    cv::Mat rgb = cv::Mat::zeros(240, 320, CV_8UC3);
    cv::Mat vis = drawDoorBoundary(rgb, ret);
    cv::imshow("door", vis);
    cv::waitKey(0);
    return 0;
}