#include "counter.h"
#include <algorithm>
#include <iostream>

// PeopleCounter 构造函数
PeopleCounter::PeopleCounter(const CounterConfig& cfg, int img_height)
    : config(cfg), image_height(img_height) {
    line_a_y = static_cast<int>(image_height * config.LINE_A_RATIO);
    line_b_y = static_cast<int>(image_height * config.LINE_B_RATIO);
}

// 检查是否穿越线条
bool PeopleCounter::isCrossingLine(float current_y, int line_y, float previous_y) {
    return (previous_y < line_y && line_y <= current_y) ||
        (previous_y > line_y && line_y >= current_y);
}

// 确定穿越方向
CrossDirection PeopleCounter::determineDirection(const TrajectoryState& state) {
    if (state.cross_order.size() < 2) {
        return CrossDirection::NONE;
    }

    const std::string& first_line = state.cross_order[0];
    const std::string& second_line = state.cross_order[1];

    // 检查时间窗口
    int time_elapsed = frame_count - state.first_cross_time;
    if (time_elapsed > config.TIME_WINDOW_FRAMES) {
        return CrossDirection::NONE;
    }

    // 判断方向
    if (first_line == "A" && second_line == "B") {
        return CrossDirection::ENTER;
    }
    else if (first_line == "B" && second_line == "A") {
        return CrossDirection::EXIT;
    }

    return CrossDirection::NONE;
}

// 记录计数事件
void PeopleCounter::recordCount(const TrajectoryState& state, const std::string& count_type) {
    CountRecord record;
    record.frame = frame_count;
    record.track_id = state.track_id;
    record.type = count_type;
    record.cross_order = state.cross_order;
    count_history.push_back(record);
}

// 清理失效轨迹状态
void PeopleCounter::cleanupStates(const std::set<int>& current_track_ids) {
    std::vector<int> ids_to_remove;
    for (const auto& pair : trajectory_states) {
        int tid = pair.first;
        if (current_track_ids.find(tid) == current_track_ids.end()) {
            ids_to_remove.push_back(tid);
        }
    }

    for (int tid : ids_to_remove) {
        trajectory_states.erase(tid);
    }
}

// 检查穿越逻辑
void PeopleCounter::checkCrossing(TrajectoryState& state, const std::tuple<float, float>& center) {
    float x = std::get<0>(center);
    float y = std::get<1>(center);

    // 检查线A穿越
    if (!state.crossed_line_a) {
        float prev_y = std::get<1>(state.last_position);
        if (isCrossingLine(y, line_a_y, prev_y)) {
            state.crossed_line_a = true;
            state.cross_order.push_back("A");
            if (state.first_cross_time == 0) {
                state.first_cross_time = frame_count;
            }
        }
    }

    // 检查线B穿越
    if (!state.crossed_line_b) {
        float prev_y = std::get<1>(state.last_position);
        if (isCrossingLine(y, line_b_y, prev_y)) {
            state.crossed_line_b = true;
            state.cross_order.push_back("B");
            if (state.first_cross_time == 0) {
                state.first_cross_time = frame_count;
            }
        }
    }

    // 判断计数
    if (state.cross_order.size() >= 2 && !state.counted) {
        CrossDirection dir = determineDirection(state);

        if (dir == CrossDirection::ENTER) {
            enter_count++;
            state.counted = true;
            recordCount(state, "ENTER");
        }
        else if (dir == CrossDirection::EXIT) {
            exit_count++;
            state.counted = true;
            recordCount(state, "EXIT");
        }
    }
}

// 更新计数器
std::tuple<int, int> PeopleCounter::update(const std::vector<Track>& tracks) {
    frame_count++;

    // 获取当前帧所有track id
    std::set<int> current_track_ids;
    for (const Track& track : tracks) {
        current_track_ids.insert(track.track_id);
    }

    // 更新每个目标状态
    for (const Track& track : tracks) {
        int track_id = track.track_id;
        auto center = track.getCenter();

        // 获取或创建轨迹状态
        if (trajectory_states.find(track_id) == trajectory_states.end()) {
            trajectory_states.emplace(track_id, TrajectoryState(track_id));
        }

        TrajectoryState& state = trajectory_states[track_id];
        state.last_position = center;

        // 检查穿越（未计数的目标）
        if (!state.counted) {
            checkCrossing(state, center);
        }
    }

    // 清理失效状态
    cleanupStates(current_track_ids);

    return { enter_count, exit_count };
}

// 获取当前计数
std::tuple<int, int> PeopleCounter::getCount() const {
    return { enter_count, exit_count };
}

// 重置计数器
void PeopleCounter::reset() {
    enter_count = 0;
    exit_count = 0;
    trajectory_states.clear();
    count_history.clear();
    frame_count = 0;
}

// 获取计数线位置
std::tuple<int, int> PeopleCounter::getLinePositions() const {
    return { line_a_y, line_b_y };
}

// 获取计数历史
const std::vector<CountRecord>& PeopleCounter::getCountHistory() const {
    return count_history;
}

// 绘制计数线
cv::Mat drawCountLines(const cv::Mat& image, int line_a_y, int line_b_y,
    const cv::Scalar& color_a, const cv::Scalar& color_b, int thickness) {
    cv::Mat result = image.clone();
    int width = result.cols;

    // 绘制线A
    cv::line(result, cv::Point(0, line_a_y), cv::Point(width, line_a_y), color_a, thickness);
    cv::putText(result, "Line A (Outside)", cv::Point(10, line_a_y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color_a, 1);

    // 绘制线B
    cv::line(result, cv::Point(0, line_b_y), cv::Point(width, line_b_y), color_b, thickness);
    cv::putText(result, "Line B (Inside)", cv::Point(10, line_b_y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color_b, 1);

    return result;
}

// 绘制计数信息
cv::Mat drawCountInfo(const cv::Mat& image, int enter_count, int exit_count,
    const cv::Scalar& color) {
    cv::Mat result = image.clone();
    std::string info_text = "Enter: " + std::to_string(enter_count) + "  Exit: " + std::to_string(exit_count);

    cv::putText(result, info_text, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);

    return result;
}

// 测试主函数
int main() {
    // 创建计数器
    PeopleCounter counter(CounterConfig(), 240);

    // 模拟跟踪目标
    Track track_enter;
    track_enter.track_id = 1;
    track_enter.bbox = cv::Rect2f(100, 30, 30, 50);  // x, y, w, h

    Track track_exit;
    track_exit.track_id = 2;
    track_exit.bbox = cv::Rect2f(150, 200, 30, 50);

    // 获取计数线位置
    auto [line_a, line_b] = counter.getLinePositions();
    std::cout << "计数线位置: A=" << line_a << ", B=" << line_b << std::endl;

    // 模拟20帧
    for (int frame = 0; frame < 20; frame++) {
        // 更新上车目标位置（从上往下）
        track_enter.bbox.y += 10;
        track_enter.bbox.height += 10;

        // 更新下车目标位置（从下往上）
        track_exit.bbox.y -= 10;
        track_exit.bbox.height -= 10;

        // 更新计数器
        std::vector<Track> tracks = { track_enter, track_exit };
        auto [enter, exit] = counter.update(tracks);

        std::cout << "Frame " << frame << ": Enter=" << enter << ", Exit=" << exit << std::endl;
    }

    // 最终计数
    auto [final_enter, final_exit] = counter.getCount();
    std::cout << "\n最终计数: 上车=" << final_enter << ", 下车=" << final_exit << std::endl;

    return 0;
}