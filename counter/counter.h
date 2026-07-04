#ifndef COUNTER_H
#define COUNTER_H

#include <vector>
#include <tuple>
#include <map>
#include <set>
#include <string>
#include <opencv2/opencv.hpp>

// 穿越方向枚举
enum class CrossDirection {
    NONE = 0,       // 未穿越
    ENTER = 1,      // 上车（外→内）
    EXIT = 2        // 下车（内→外）
};

// 计数配置结构体
struct CounterConfig {
    float LINE_A_RATIO = 0.3f;    // 外侧线比例
    float LINE_B_RATIO = 0.7f;    // 内侧线比例
    int TIME_WINDOW_FRAMES = 30;  // 时间窗口（帧）
};

// 跟踪目标结构体（模拟tracker::Track）
struct Track {
    int track_id;
    cv::Rect2f bbox;  // 边界框 (x, y, width, height)

    // 获取中心点
    std::tuple<float, float> getCenter() const {
        float x = bbox.x + bbox.width / 2;
        float y = bbox.y + bbox.height / 2;
        return { x, y };
    }
};

// 轨迹状态结构体
struct TrajectoryState {
    int track_id;
    bool crossed_line_a = false;
    bool crossed_line_b = false;
    std::vector<std::string> cross_order;
    int first_cross_time = 0;
    std::tuple<float, float> last_position = { 0.0f, 0.0f };
    bool counted = false;

    TrajectoryState(int id) : track_id(id) {}
};

// 计数历史记录结构体
struct CountRecord {
    int frame;
    int track_id;
    std::string type;
    std::vector<std::string> cross_order;
};

// 客流计数器类
class PeopleCounter {
private:
    CounterConfig config;
    int image_height;
    int line_a_y;
    int line_b_y;
    int enter_count = 0;
    int exit_count = 0;
    std::map<int, TrajectoryState> trajectory_states;
    int frame_count = 0;
    std::vector<CountRecord> count_history;

    // 检查是否穿越某条线
    bool isCrossingLine(float current_y, int line_y, float previous_y);

    // 确定穿越方向
    CrossDirection determineDirection(const TrajectoryState& state);

    // 记录计数事件
    void recordCount(const TrajectoryState& state, const std::string& count_type);

    // 清理已消失的目标状态
    void cleanupStates(const std::set<int>& current_track_ids);

    // 检查穿越逻辑
    void checkCrossing(TrajectoryState& state, const std::tuple<float, float>& center);

public:
    // 构造函数
    PeopleCounter(const CounterConfig& cfg = CounterConfig(), int img_height = 240);

    // 更新计数器
    std::tuple<int, int> update(const std::vector<Track>& tracks);

    // 获取当前计数
    std::tuple<int, int> getCount() const;

    // 重置计数器
    void reset();

    // 获取计数线位置
    std::tuple<int, int> getLinePositions() const;

    // 获取计数历史（调试用）
    const std::vector<CountRecord>& getCountHistory() const;
};

// 可视化函数
cv::Mat drawCountLines(const cv::Mat& image, int line_a_y, int line_b_y,
    const cv::Scalar& color_a = cv::Scalar(0, 255, 255),
    const cv::Scalar& color_b = cv::Scalar(255, 0, 255),
    int thickness = 2);

cv::Mat drawCountInfo(const cv::Mat& image, int enter_count, int exit_count,
    const cv::Scalar& color = cv::Scalar(255, 255, 255));

#endif // COUNTER_H