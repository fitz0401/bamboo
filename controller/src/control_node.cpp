// Joint Impedance control

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <msgpack.hpp>
#include <zmq.hpp>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include <Eigen/Dense>

#include "bamboo_messages.h"
#include "controllers/joint_impedance_controller.h"
#include "interpolators/joint_interpolator.h"

// Global flag for signal handling
std::atomic<bool> global_shutdown{false};

// Global exception handling
std::mutex exception_mutex;
std::exception_ptr thread_exception_ptr = nullptr;

void setThreadException(std::exception_ptr ex) {
  std::lock_guard<std::mutex> lock(exception_mutex);
  if (!thread_exception_ptr) {
    thread_exception_ptr = ex;
    global_shutdown = true;
  }
}

std::exception_ptr getThreadException() {
  std::lock_guard<std::mutex> lock(exception_mutex);
  return thread_exception_ptr;
}

namespace {

using Vector7d = Eigen::Matrix<double, 7, 1>;
using SteadyClock = std::chrono::steady_clock;

static_assert(std::atomic<double>::is_always_lock_free,
              "Streaming control requires lock-free double atomics");

constexpr std::array<double, 7> kFr3JointLower = {
    -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508};
constexpr std::array<double, 7> kFr3JointUpper = {
    2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508};
constexpr std::array<double, 7> kPandaJointLower = {
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
constexpr std::array<double, 7> kPandaJointUpper = {
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
constexpr double kJointMargin = 0.08;
constexpr double kHardMaxStreamVelocity = 0.50;
constexpr double kHardMaxStreamAcceleration = 2.0;
constexpr int kHardMaxWatchdogMs = 500;
constexpr double kMaxTrackingError = 0.03;

int64_t steadyNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             SteadyClock::now().time_since_epoch())
      .count();
}

} // namespace

// Signal handler for graceful shutdown
void signalHandler(int signal) {
  if (signal == SIGINT || signal == SIGTERM || signal == SIGHUP) {
    global_shutdown = true;
  }
}

// Server implementation
class BambooControlServer {
private:
  franka::Robot *robot_;
  franka::Model *model_;
  franka::Gripper *gripper_;
  bamboo::controllers::JointImpedanceController *controller_;
  bamboo::interpolators::JointInterpolator *interpolator_;

  std::atomic<bool> control_running_{false};
  std::atomic<bool> joint_limit_hit_{false};
  const std::array<double, 7> joint_lower_;
  const std::array<double, 7> joint_upper_;

  // Persistent streaming control state. All callback-shared values are atomic
  // so the 1 kHz real-time loop never waits on the ZMQ thread.
  std::array<std::atomic<double>, 7> stream_target_velocity_;
  std::array<std::atomic<double>, 7> state_q_;
  std::array<std::atomic<double>, 7> state_dq_;
  std::array<std::atomic<double>, 7> state_tau_;
  std::array<std::atomic<double>, 16> state_pose_;
  std::atomic<double> state_time_{0.0};
  std::atomic<int64_t> stream_last_update_ns_{0};
  std::atomic<int64_t> stream_stop_request_ns_{0};
  std::atomic<bool> stream_running_{false};
  std::atomic<bool> stream_ready_{false};
  std::atomic<bool> stream_stop_requested_{false};
  std::atomic<bool> stream_failed_{false};
  std::thread stream_thread_;
  std::mutex stream_error_mutex_;
  std::string stream_error_message_;
  int stream_watchdog_ms_{250};
  double stream_max_velocity_{0.35};
  double stream_max_acceleration_{1.5};

  // Control parameters
  const int traj_rate_ = 500; // Hz
  const double max_time = 1.0;
  const bool log_err_ = true;

  Eigen::Matrix<double, 7, 1> q_current_;
  Eigen::Matrix<double, 7, 1> q_goal_;

  // For acceleration computation via finite differencing
  Eigen::Matrix<double, 7, 1> velocity_cmd_prev_;
  Eigen::Matrix<double, 7, 1> a_cmd_latest_;

  // Low-pass filter frequency for acceleration
  const double diff_low_pass_freq_ = 30.0; // Hz

public:
  BambooControlServer(franka::Robot *robot, franka::Model *model,
                      franka::Gripper *gripper,
                      bamboo::controllers::JointImpedanceController *controller,
                      bamboo::interpolators::JointInterpolator *interpolator,
                      const std::array<double, 7> &joint_lower,
                      const std::array<double, 7> &joint_upper)
      : robot_(robot), model_(model), gripper_(gripper),
        controller_(controller), interpolator_(interpolator),
        joint_lower_(joint_lower), joint_upper_(joint_upper) {

    // Get initial robot state
    franka::RobotState init_state = robot_->readOnce();
    cacheRobotState(init_state);
    q_current_ = Eigen::VectorXd::Map(init_state.q.data(), 7);
    q_goal_ = q_current_;

    // Initialize velocity and acceleration tracking
    velocity_cmd_prev_.setZero();
    a_cmd_latest_.setZero();
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }

    std::cout << "Initial joint positions: " << q_current_.transpose()
              << std::endl;
  }

  ~BambooControlServer() {
    try {
      StopStreaming();
    } catch (const std::exception &e) {
      std::cerr << "[STREAM] Error during shutdown: " << e.what() << std::endl;
    }
  }

  bamboo_msgs::RobotState GetRobotState() {
    try {
      // readOnce cannot run concurrently with robot.control(). During a stream,
      // return the lock-free state cache populated by the control callback.
      if (!stream_running_.load()) {
        cacheRobotState(robot_->readOnce());
      }

      bamboo_msgs::RobotState state_msg;

      // Add joint positions, velocities, and torques
      state_msg.q.resize(7);
      state_msg.dq.resize(7);
      state_msg.tau_J.resize(7);
      for (size_t i = 0; i < 7; ++i) {
        state_msg.q[i] = state_q_[i].load();
        state_msg.dq[i] = state_dq_[i].load();
        state_msg.tau_J[i] = state_tau_[i].load();
      }

      // Add end-effector pose (4x4 transformation matrix: O_T_EE)
      state_msg.O_T_EE.resize(16);
      for (size_t i = 0; i < 16; ++i) {
        state_msg.O_T_EE[i] = state_pose_[i].load();
      }

      // Add timing information
      state_msg.time_sec = state_time_.load();

      return state_msg;
    } catch (const franka::Exception &e) {
      throw std::runtime_error(std::string("Franka exception: ") + e.what());
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Exception: ") + e.what());
    }
  }

  void StartStreaming(int watchdog_ms, double max_velocity,
                      double max_acceleration) {
    if (stream_running_.load()) {
      // Reconnecting clients always inherit a stopped stream.
      SetStreamVelocity(std::vector<double>(7, 0.0));
      return;
    }
    if (control_running_.load()) {
      throw std::runtime_error("Another control loop is already running");
    }
    if (stream_thread_.joinable()) {
      stream_thread_.join();
    }

    stream_watchdog_ms_ = std::clamp(watchdog_ms, 50, kHardMaxWatchdogMs);
    stream_max_velocity_ =
        std::clamp(max_velocity, 0.01, kHardMaxStreamVelocity);
    stream_max_acceleration_ =
        std::clamp(max_acceleration, 0.05, kHardMaxStreamAcceleration);

    const franka::RobotState initial_state = robot_->readOnce();
    cacheRobotState(initial_state);
    q_current_ = Eigen::Map<const Vector7d>(initial_state.q.data());
    q_goal_ = q_current_;
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
    stream_stop_request_ns_.store(0);
    stream_stop_requested_.store(false);
    stream_failed_.store(false);
    stream_ready_.store(false);
    setStreamError("");
    control_running_.store(true);
    stream_running_.store(true);
    stream_thread_ = std::thread(&BambooControlServer::streamControlLoop, this);

    const auto deadline = SteadyClock::now() + std::chrono::seconds(2);
    while (!stream_ready_.load() && stream_running_.load() &&
           SteadyClock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!stream_ready_.load()) {
      stream_stop_requested_.store(true);
      if (stream_thread_.joinable()) {
        stream_thread_.join();
      }
      const std::string message = getStreamError();
      throw std::runtime_error(message.empty() ? "Stream failed to start"
                                               : message);
    }
    std::cout << "[STREAM] Started (watchdog=" << stream_watchdog_ms_
              << " ms, max_velocity=" << stream_max_velocity_
              << " rad/s, max_acceleration=" << stream_max_acceleration_
              << " rad/s^2)" << std::endl;
  }

  void SetStreamVelocity(const std::vector<double> &velocity) {
    if (!stream_running_.load() || !stream_ready_.load()) {
      throw std::runtime_error("Streaming control is not running");
    }
    if (velocity.size() != 7) {
      throw std::runtime_error("Velocity must contain seven values");
    }
    for (size_t i = 0; i < velocity.size(); ++i) {
      if (!std::isfinite(velocity[i])) {
        throw std::runtime_error("Velocity values must be finite");
      }
      if (std::abs(velocity[i]) > stream_max_velocity_ + 1e-9) {
        throw std::runtime_error("Velocity exceeds negotiated stream limit");
      }
    }
    for (size_t i = 0; i < velocity.size(); ++i) {
      stream_target_velocity_[i].store(velocity[i]);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
  }

  void StopStreaming() {
    if (!stream_running_.load() && !stream_thread_.joinable()) {
      return;
    }
    for (auto &value : stream_target_velocity_) {
      value.store(0.0);
    }
    stream_last_update_ns_.store(steadyNowNanoseconds());
    stream_stop_request_ns_.store(steadyNowNanoseconds());
    stream_stop_requested_.store(true);
    if (stream_thread_.joinable()) {
      stream_thread_.join();
    }
    if (stream_failed_.load()) {
      throw std::runtime_error(getStreamError());
    }
    std::cout << "[STREAM] Stopped" << std::endl;
  }

  bool ExecuteJointImpedanceTrajectory(
      const bamboo_msgs::TrajectoryRequest &request) {
    std::cout << "[SERVER] Received trajectory with "
              << request.waypoints.size() << " waypoints" << std::endl;

    try {
      // Check if already running
      if (control_running_.load()) {
        throw std::runtime_error("Control loop already running");
      }

      if (request.waypoints.empty()) {
        throw std::runtime_error("Empty trajectory");
      }

      // Parse and prepare all waypoints
      std::vector<Eigen::Matrix<double, 7, 1>> trajectory_goals;
      std::vector<Eigen::Matrix<double, 7, 1>> trajectory_velocities;
      std::vector<double> trajectory_durations;
      std::vector<std::optional<std::array<double, 7>>> trajectory_kp;
      std::vector<std::optional<std::array<double, 7>>> trajectory_kd;

      for (size_t i = 0; i < request.waypoints.size(); ++i) {
        const bamboo_msgs::TimedWaypoint &waypoint = request.waypoints[i];

        // Validate goal has 7 values
        if (waypoint.q_goal.size() != 7) {
          throw std::runtime_error("Joint configuration must have 7 values");
        }

        // Get waypoint duration
        double waypoint_duration;
        if (waypoint.duration > 0) {
          waypoint_duration = waypoint.duration;
        } else if (request.default_duration > 0) {
          waypoint_duration = request.default_duration;
        } else {
          waypoint_duration = 1.0; // Fallback default
        }

        // Get waypoint velocity
        Eigen::Matrix<double, 7, 1> waypoint_velocity =
            Eigen::Matrix<double, 7, 1>::Zero();
        if (waypoint.velocity.size() == 7) {
          // Use waypoint-specific velocity if provided
          for (int j = 0; j < 7; ++j) {
            waypoint_velocity(j) = waypoint.velocity[j];
          }
        } else if (request.default_velocity.size() == 7) {
          // Use default velocity if provided
          for (int j = 0; j < 7; ++j) {
            waypoint_velocity(j) = request.default_velocity[j];
          }
        }
        // If neither is provided, waypoint_velocity remains zero

        // Get waypoint kp (optional)
        std::optional<std::array<double, 7>> waypoint_kp = std::nullopt;
        if (waypoint.kp.size() == 7) {
          std::array<double, 7> kp_array;
          for (size_t j = 0; j < 7; ++j) {
            kp_array[j] = waypoint.kp[j];
          }
          waypoint_kp = kp_array;
        }

        // Get waypoint kd (optional)
        std::optional<std::array<double, 7>> waypoint_kd = std::nullopt;
        if (waypoint.kd.size() == 7) {
          std::array<double, 7> kd_array;
          for (size_t j = 0; j < 7; ++j) {
            kd_array[j] = waypoint.kd[j];
          }
          waypoint_kd = kd_array;
        }

        // Check for termination
        if (global_shutdown) {
          std::cout << "[SERVER] Termination requested" << std::endl;
          break;
        }

        // Store goal position
        Eigen::Matrix<double, 7, 1> goal;
        for (int j = 0; j < 7; ++j) {
          goal(j) = waypoint.q_goal[j];
        }

        trajectory_goals.push_back(goal);
        trajectory_velocities.push_back(waypoint_velocity);
        trajectory_durations.push_back(waypoint_duration);
        trajectory_kp.push_back(waypoint_kp);
        trajectory_kd.push_back(waypoint_kd);
      }

      // Execute entire trajectory in single control call
      bool success =
          executeTrajectory(trajectory_goals, trajectory_velocities,
                            trajectory_durations, trajectory_kp, trajectory_kd);
      if (!success) {
        throw std::runtime_error("Trajectory execution failed");
      }

      std::cout << "[SERVER] Trajectory completed successfully" << std::endl;
      return true;

    } catch (const franka::ControlException &e) {
      std::cerr << "[SERVER] Control exception: " << e.what() << std::endl;
      throw;
    } catch (const std::exception &e) {
      std::cerr << "[SERVER] Exception: " << e.what() << std::endl;
      throw;
    }
  }

  bool OpenGripper(double width, double speed) {
    std::cout << "[SERVER] Opening gripper to width: " << width
              << "m, speed: " << speed << "m/s" << std::endl;
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }
      return gripper_->move(width, speed);
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception during gripper open: " << e.what()
                << std::endl;
      throw;
    }
  }

  bool CloseGripper(double speed, double force) {
    std::cout << "[SERVER] Closing gripper: speed=" << speed
              << "m/s, force=" << force << "N" << std::endl;
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }
      // epsilon is set such that the grasp succeeds if the gripper_width is
      // b/w (0.0 - 0.08, 0.0 + 0.08). this way we don't throw when
      // grasping large objects
      return gripper_->grasp(0.0, speed, force, 0.08, 0.08);
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception during gripper close: "
                << e.what() << std::endl;
      throw;
    }
  }

  std::map<std::string, double> GetGripperState() {
    try {
      if (!gripper_) {
        throw std::runtime_error("Gripper not initialized");
      }

      franka::GripperState state = gripper_->readOnce();
      std::map<std::string, double> result;
      result["width"] = state.width;
      result["max_width"] = state.max_width;
      result["is_grasped"] = state.is_grasped ? 1.0 : 0.0;
      // open_gripper and close_gripper are blocking, so the gripper is never
      // mid-motion when queried.
      result["is_moving"] = 0.0;
      result["temperature"] = state.temperature;

      return result;
    } catch (const franka::Exception &e) {
      std::cerr << "[SERVER] Franka exception reading gripper state: "
                << e.what() << std::endl;
      throw;
    }
  }

private:
  void cacheRobotState(const franka::RobotState &state) {
    for (size_t i = 0; i < 7; ++i) {
      state_q_[i].store(state.q[i], std::memory_order_relaxed);
      state_dq_[i].store(state.dq[i], std::memory_order_relaxed);
      state_tau_[i].store(state.tau_J[i], std::memory_order_relaxed);
    }
    for (size_t i = 0; i < 16; ++i) {
      state_pose_[i].store(state.O_T_EE[i], std::memory_order_relaxed);
    }
    state_time_.store(state.time.toSec(), std::memory_order_relaxed);
  }

  void setStreamError(const std::string &message) {
    std::lock_guard<std::mutex> lock(stream_error_mutex_);
    stream_error_message_ = message;
  }

  std::string getStreamError() {
    std::lock_guard<std::mutex> lock(stream_error_mutex_);
    return stream_error_message_;
  }

  void streamControlLoop() {
    try {
      Vector7d desired_position;
      Vector7d desired_velocity = Vector7d::Zero();
      for (size_t i = 0; i < 7; ++i) {
        desired_position[static_cast<int>(i)] = state_q_[i].load();
      }
      controller_->RestoreDefaultGains();
      stream_ready_.store(true);

      auto control_callback = [&](const franka::RobotState &robot_state,
                                  franka::Duration period) -> franka::Torques {
        cacheRobotState(robot_state);
        q_current_ = Eigen::Map<const Vector7d>(robot_state.q.data());
        const double dt = std::clamp(period.toSec(), 0.0001, 0.01);
        Vector7d requested_velocity;
        const int64_t command_age_ns =
            steadyNowNanoseconds() - stream_last_update_ns_.load();
        const bool watchdog_expired =
            command_age_ns >
            static_cast<int64_t>(stream_watchdog_ms_) * 1000000;
        const bool ending =
            stream_stop_requested_.load() || global_shutdown.load();
        if (ending && stream_stop_request_ns_.load() == 0) {
          int64_t expected = 0;
          stream_stop_request_ns_.compare_exchange_strong(
              expected, steadyNowNanoseconds());
        }

        for (size_t i = 0; i < 7; ++i) {
          requested_velocity[static_cast<int>(i)] =
              (watchdog_expired || ending) ? 0.0
                                           : stream_target_velocity_[i].load();
        }

        const double max_velocity_delta = stream_max_acceleration_ * dt;
        for (int i = 0; i < 7; ++i) {
          const double difference = requested_velocity[i] - desired_velocity[i];
          desired_velocity[i] +=
              std::clamp(difference, -max_velocity_delta, max_velocity_delta);

          const double next = desired_position[i] + desired_velocity[i] * dt;
          const double lower = joint_lower_[i] + kJointMargin;
          const double upper = joint_upper_[i] - kJointMargin;
          if (next < lower || next > upper) {
            desired_velocity[i] = 0.0;
            desired_position[i] = std::clamp(next, lower, upper);
          } else {
            desired_position[i] = next;
          }

          // Bound reference wind-up if contact prevents target tracking.
          desired_position[i] = std::clamp(
              desired_position[i], robot_state.q[i] - kMaxTrackingError,
              robot_state.q[i] + kMaxTrackingError);
        }

        const bamboo::controllers::ControllerResult result = controller_->Step(
            robot_state, desired_position, desired_velocity, Vector7d::Zero());
        const std::array<double, 7> torques = franka::limitRate(
            franka::kMaxTorqueRate, result.torques, robot_state.tau_J_d);
        if (result.torque_limit_violated) {
          stream_failed_.store(true);
          setStreamError("Torque limit reached during streaming control");
          return franka::MotionFinished(franka::Torques(torques));
        }

        const Vector7d measured_velocity =
            Eigen::Map<const Vector7d>(robot_state.dq.data());
        const double stop_age_seconds =
            static_cast<double>(steadyNowNanoseconds() -
                                stream_stop_request_ns_.load()) /
            1e9;
        if (ending && desired_velocity.norm() < 1e-4 &&
            (measured_velocity.norm() < 0.01 || stop_age_seconds > 2.0)) {
          return franka::MotionFinished(franka::Torques(torques));
        }
        return franka::Torques(torques);
      };

      robot_->control(control_callback);
    } catch (const franka::Exception &e) {
      stream_failed_.store(true);
      setStreamError(std::string("Franka streaming control error: ") +
                     e.what());
    } catch (const std::exception &e) {
      stream_failed_.store(true);
      setStreamError(std::string("Streaming control error: ") + e.what());
    }
    stream_ready_.store(false);
    stream_running_.store(false);
    control_running_.store(false);
  }

  bool executeTrajectory(
      const std::vector<Eigen::Matrix<double, 7, 1>> &goals,
      const std::vector<Eigen::Matrix<double, 7, 1>> &velocities,
      const std::vector<double> &durations,
      const std::vector<std::optional<std::array<double, 7>>> &kp_per_waypoint,
      const std::vector<std::optional<std::array<double, 7>>>
          &kd_per_waypoint) {
    if (goals.empty())
      return false;

    control_running_ = true;
    joint_limit_hit_ = false;
    double control_time = 0.0;

    // Reset velocity and acceleration tracking for new trajectory
    velocity_cmd_prev_.setZero();
    a_cmd_latest_.setZero();

    // Current waypoint tracking
    std::size_t current_waypoint = 0;
    double waypoint_start_time = 0.0;

    // Max joint error tracking (L1 norm across waypoint final errors)
    double max_joint_error_rad = 0.0;

    // Max end-effector error tracking
    double max_ee_position_error_m = 0.0;
    double max_ee_orientation_error_rad = 0.0;

    // Final waypoint end-effector error tracking
    double final_ee_position_error_m = 0.0;
    double final_ee_orientation_error_rad = 0.0;

    // Initialize first waypoint
    q_goal_ = goals[0];
    Eigen::Matrix<double, 7, 1> velocity_start =
        Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 7, 1> velocity_goal =
        velocities.empty() ? Eigen::Matrix<double, 7, 1>::Zero()
                           : velocities[0];
    interpolator_->Reset(control_time, q_current_, q_goal_, velocity_start,
                         velocity_goal, traj_rate_, durations[0]);

    // Set gains for first waypoint (or restore to defaults if not specified)
    if (kp_per_waypoint[0].has_value() && kd_per_waypoint[0].has_value()) {
      controller_->SetGains(kp_per_waypoint[0].value(),
                            kd_per_waypoint[0].value());
    } else {
      controller_->RestoreDefaultGains();
    }

    std::cout << "[CONTROL] Starting trajectory with " << goals.size()
              << " waypoints" << std::endl;

    // Single control callback for entire trajectory
    auto control_callback = [&](const franka::RobotState &robot_state,
                                franka::Duration period) -> franka::Torques {
      try {
        // Get time step
        const double dt = period.toSec();

        // Update time
        control_time += dt;

        // Update current position
        q_current_ = Eigen::VectorXd::Map(robot_state.q.data(), 7);

        // Check if current waypoint is complete
        double waypoint_elapsed = control_time - waypoint_start_time;
        if (waypoint_elapsed >= durations[current_waypoint]) {
          if (log_err_) {
            // Log joint error for waypoint that timed out
            Eigen::Matrix<double, 7, 1> waypoint_error = q_goal_ - q_current_;
            double waypoint_final_error_rad = waypoint_error.cwiseAbs().sum();
            // Update max error across all waypoints
            if (waypoint_final_error_rad > max_joint_error_rad) {
              max_joint_error_rad = waypoint_final_error_rad;
            }

            // Calculate end-effector position and orientation errors for
            // completed waypoint Get desired EE pose from goal joint angles
            std::array<double, 7> q_goal_array;
            Eigen::VectorXd::Map(&q_goal_array[0], 7) = q_goal_;

            // Create a temporary robot state with goal joint positions
            franka::RobotState temp_state = robot_state;
            temp_state.q = q_goal_array;

            std::array<double, 16> desired_ee_pose_array =
                model_->pose(franka::Frame::kEndEffector, temp_state);
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                desired_ee_pose(desired_ee_pose_array.data());

            // Get current EE pose from robot state
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                current_ee_pose(robot_state.O_T_EE.data());

            // Calculate position error (translation part)
            Eigen::Vector3d desired_position =
                desired_ee_pose.block<3, 1>(0, 3);
            Eigen::Vector3d current_position =
                current_ee_pose.block<3, 1>(0, 3);
            double current_ee_position_error =
                (desired_position - current_position).norm();
            if (current_ee_position_error > max_ee_position_error_m) {
              max_ee_position_error_m = current_ee_position_error;
            }

            // Calculate orientation error (rotation part)
            Eigen::Matrix3d desired_rotation =
                desired_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d current_rotation =
                current_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d rotation_error =
                desired_rotation * current_rotation.transpose();

            // Convert rotation matrix to angle-axis to get scalar error
            Eigen::AngleAxisd angle_axis(rotation_error);
            double current_ee_orientation_error = std::abs(angle_axis.angle());
            if (current_ee_orientation_error > max_ee_orientation_error_rad) {
              max_ee_orientation_error_rad = current_ee_orientation_error;
            }
          }

          // Move to next waypoint
          current_waypoint++;

          if (current_waypoint >= goals.size()) {
            // If still moving, continue with current control to let robot
            // settle
            current_waypoint = goals.size() - 1; // Stay on last waypoint
          } else {
            // Setup next waypoint
            waypoint_start_time = control_time;
            q_goal_ = goals[current_waypoint];
            Eigen::Matrix<double, 7, 1> velocity_prev =
                (current_waypoint > 0 &&
                 current_waypoint - 1 < velocities.size())
                    ? velocities[current_waypoint - 1]
                    : Eigen::Matrix<double, 7, 1>::Zero();
            Eigen::Matrix<double, 7, 1> velocity_curr =
                (current_waypoint < velocities.size())
                    ? velocities[current_waypoint]
                    : Eigen::Matrix<double, 7, 1>::Zero();
            interpolator_->Reset(control_time, q_current_, q_goal_,
                                 velocity_prev, velocity_curr, traj_rate_,
                                 durations[current_waypoint]);

            // Set gains for new waypoint (or restore to defaults if not
            // specified)
            if (kp_per_waypoint[current_waypoint].has_value() &&
                kd_per_waypoint[current_waypoint].has_value()) {
              controller_->SetGains(kp_per_waypoint[current_waypoint].value(),
                                    kd_per_waypoint[current_waypoint].value());
            } else {
              controller_->RestoreDefaultGains();
            }
          }
        }

        // Get interpolated desired position and velocity for current waypoint
        Eigen::Matrix<double, 7, 1> q_desired;
        Eigen::Matrix<double, 7, 1> dq_desired;
        interpolator_->GetNextStep(control_time, q_desired, dq_desired);

        // Compute desired acceleration via finite differencing
        Eigen::Matrix<double, 7, 1> ddq_desired =
            Eigen::Matrix<double, 7, 1>::Zero();
        if (dt > 0.0) {
          // Compute raw acceleration from velocity difference
          Eigen::Matrix<double, 7, 1> a_cmd_raw =
              (dq_desired - velocity_cmd_prev_) / dt;

          // Apply low-pass filter
          for (int i = 0; i < 7; ++i) {
            a_cmd_latest_[i] = franka::lowpassFilter(
                dt, a_cmd_raw[i], a_cmd_latest_[i], diff_low_pass_freq_);
          }
          ddq_desired = a_cmd_latest_;

          // Update previous velocity for next iteration
          velocity_cmd_prev_ = dq_desired;
        }

        // Compute control torques
        bamboo::controllers::ControllerResult result =
            controller_->Step(robot_state, q_desired, dq_desired, ddq_desired);

        // Check for torque limit violation
        if (result.torque_limit_violated) {
          joint_limit_hit_ = true;
          std::cout
              << "[CONTROL] Torque limit violated - ending trajectory early"
              << std::endl;
          std::array<double, 7> zero_torques = {0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0};
          return franka::MotionFinished(franka::Torques(zero_torques));
        }

        // Apply rate limiting
        std::array<double, 7> tau_d_rate_limited = franka::limitRate(
            franka::kMaxTorqueRate, result.torques, robot_state.tau_J_d);

        // Check if all waypoints completed and robot has stopped
        if (current_waypoint >= goals.size() - 1 &&
            waypoint_elapsed >= durations[current_waypoint]) {
          Eigen::VectorXd dq_current =
              Eigen::VectorXd::Map(robot_state.dq.data(), 7);
          double velocity_norm = dq_current.norm();

          if (velocity_norm <
              0.01) { // Robot has stopped (threshold: 0.01 rad/s)
            // Calculate final waypoint errors (robot vs last goal) - robot is
            // now still
            Eigen::Matrix<double, 7, 1> final_goal = goals.back();
            std::array<double, 7> final_goal_array;
            Eigen::VectorXd::Map(&final_goal_array[0], 7) = final_goal;

            // Get desired EE pose for final waypoint
            franka::RobotState final_temp_state = robot_state;
            final_temp_state.q = final_goal_array;
            std::array<double, 16> final_desired_ee_pose_array =
                model_->pose(franka::Frame::kEndEffector, final_temp_state);
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                final_desired_ee_pose(final_desired_ee_pose_array.data());

            // Get current EE pose from robot state
            Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::ColMajor>>
                current_ee_pose(robot_state.O_T_EE.data());

            // Calculate final position error
            Eigen::Vector3d final_desired_position =
                final_desired_ee_pose.block<3, 1>(0, 3);
            Eigen::Vector3d current_position =
                current_ee_pose.block<3, 1>(0, 3);
            final_ee_position_error_m =
                (final_desired_position - current_position).norm();

            // Calculate final orientation error
            Eigen::Matrix3d final_desired_rotation =
                final_desired_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d current_rotation =
                current_ee_pose.block<3, 3>(0, 0);
            Eigen::Matrix3d final_rotation_error =
                final_desired_rotation * current_rotation.transpose();
            Eigen::AngleAxisd final_angle_axis(final_rotation_error);
            final_ee_orientation_error_rad = std::abs(final_angle_axis.angle());

            std::cout << "[CONTROL] All waypoints completed, robot stopped"
                      << std::endl;
            return franka::MotionFinished(franka::Torques(tau_d_rate_limited));
          }
        }

        // Check if we should stop
        if (!control_running_ || global_shutdown) {
          return franka::MotionFinished(franka::Torques(tau_d_rate_limited));
        }

        return franka::Torques(tau_d_rate_limited);
      } catch (const std::exception &e) {
        std::cerr << "[CONTROL_CALLBACK] Exception: " << e.what() << std::endl;
        std::array<double, 7> zero_torques = {0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0};
        return franka::MotionFinished(franka::Torques(zero_torques));
      }
    };

    try {
      // Execute control
      robot_->control(control_callback);
      control_running_ = false;

      // Check if trajectory failed due to joint limit violation
      if (joint_limit_hit_) {
        std::cerr
            << "[TRAJECTORY] Trajectory failed due to joint limit violation"
            << std::endl;
        return false;
      }

      if (log_err_) {
        // Print max joint error across all waypoint final errors
        double max_joint_error_deg = max_joint_error_rad * 180.0 / M_PI;
        std::cout << "[CONTROL] Max sum of joint errors during trajectory: "
                  << std::fixed << std::setprecision(2) << max_joint_error_deg
                  << " degrees" << std::endl;

        // Print end-effector error metrics
        double max_ee_orientation_error_deg =
            max_ee_orientation_error_rad * 180.0 / M_PI;
        std::cout << "[CONTROL] Max EE position error during trajectory: "
                  << std::fixed << std::setprecision(4)
                  << max_ee_position_error_m * 1000.0 << " mm" << std::endl;
        std::cout << "[CONTROL] Max EE orientation error during trajectory: "
                  << std::fixed << std::setprecision(2)
                  << max_ee_orientation_error_deg << " degrees" << std::endl;
      }

      // Print final waypoint errors
      double final_ee_orientation_error_deg =
          final_ee_orientation_error_rad * 180.0 / M_PI;
      std::cout << "[CONTROL] Final EE position error (vs last waypoint): "
                << std::fixed << std::setprecision(4)
                << final_ee_position_error_m * 1000.0 << " mm" << std::endl;
      std::cout << "[CONTROL] Final EE orientation error (vs last waypoint): "
                << std::fixed << std::setprecision(2)
                << final_ee_orientation_error_deg << " degrees" << std::endl;

      return true;
    } catch (const franka::ControlException &e) {
      std::cerr << "[TRAJECTORY] Control exception: " << e.what() << std::endl;
      std::cerr << "[TRAJECTORY] Resetting Error State" << std::endl;
      std::cerr << "[Trajectory] If errors persist, you may have to restart "
                   "the controller by ending this tmux session "
                   "and running RunBambooController"
                << std::endl;
      robot_->automaticErrorRecovery();
      control_running_ = false;
      return false;
    }
  }
};

// Message parsing helpers
using RequestMap = std::map<std::string, msgpack::object>;

template <typename T>
T valueOr(const RequestMap &values, const std::string &key, const T &fallback) {
  const auto it = values.find(key);
  if (it == values.end()) {
    return fallback;
  }
  T value;
  it->second.convert(value);
  return value;
}

RequestMap getRequestData(const RequestMap &request_map) {
  const auto it = request_map.find("data");
  if (it == request_map.end()) {
    return {};
  }
  RequestMap data;
  it->second.convert(data);
  return data;
}

std::string parseCommand(const RequestMap &request_map) {
  std::string command;
  auto it = request_map.find("command");
  if (it != request_map.end()) {
    it->second.convert(command);
  }
  return command;
}

msgpack::sbuffer handleGetCapabilities() {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(3);
  packer.pack("success");
  packer.pack(true);
  packer.pack("protocol_version");
  packer.pack(1);
  packer.pack("features");
  packer.pack(std::vector<std::string>{"stream_joint_velocity",
                                       "cached_robot_state", "watchdog"});
  return response_buf;
}

msgpack::sbuffer handleStartStream(BambooControlServer &server,
                                   const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  server.StartStreaming(valueOr<int>(data, "watchdog_ms", 250),
                        valueOr<double>(data, "max_joint_velocity", 0.35),
                        valueOr<double>(data, "max_joint_acceleration", 1.5));

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleStreamJointVelocity(BambooControlServer &server,
                                           const RequestMap &request_map) {
  const RequestMap data = getRequestData(request_map);
  server.SetStreamVelocity(valueOr<std::vector<double>>(data, "velocity", {}));

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleStopStream(BambooControlServer &server) {
  server.StopStreaming();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);
  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));
  return response_buf;
}

msgpack::sbuffer handleGetRobotState(BambooControlServer &server) {
  bamboo_msgs::RobotState state = server.GetRobotState();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("data");
  packer.pack(state);

  return response_buf;
}

msgpack::sbuffer handleExecuteTrajectory(
    BambooControlServer &server,
    const std::map<std::string, msgpack::object> &request_map) {
  bamboo_msgs::TrajectoryRequest traj_req;
  auto it = request_map.find("data");
  if (it != request_map.end()) {
    it->second.convert(traj_req);
  }

  bool success = server.ExecuteJointImpedanceTrajectory(traj_req);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(success);
  packer.pack("error");
  if (!success) {
    packer.pack(
        std::string("Joint limit violated during trajectory execution"));
  } else {
    packer.pack(std::string(""));
  }

  return response_buf;
}

msgpack::sbuffer
handleOpenGripper(BambooControlServer &server,
                  const std::map<std::string, msgpack::object> &request_map) {
  // Parse gripper parameters
  double width = 0.08; // Default max width for Franka Hand
  double speed = 0.05; // Default speed

  auto it_width = request_map.find("width");
  if (it_width != request_map.end()) {
    it_width->second.convert(width);
  }

  auto it_speed = request_map.find("speed");
  if (it_speed != request_map.end()) {
    it_speed->second.convert(speed);
  }

  bool success = server.OpenGripper(width, speed);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(1);
  packer.pack("success");
  packer.pack(success);

  return response_buf;
}

msgpack::sbuffer
handleCloseGripper(BambooControlServer &server,
                   const std::map<std::string, msgpack::object> &request_map) {
  // Parse gripper parameters
  double speed = 0.05; // Default speed
  double force = 20.0; // Default force in Newtons

  auto it_speed = request_map.find("speed");
  if (it_speed != request_map.end()) {
    it_speed->second.convert(speed);
  }

  auto it_force = request_map.find("force");
  if (it_force != request_map.end()) {
    double force_normalized;
    it_force->second.convert(force_normalized);
    // Convert from 0-1 range to Newtons (5-70N)
    force = 5.0 + force_normalized * 65.0;
  }

  bool success = server.CloseGripper(speed, force);

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(1);
  packer.pack("success");
  packer.pack(success);

  return response_buf;
}

msgpack::sbuffer handleGetGripperState(BambooControlServer &server) {
  std::map<std::string, double> state = server.GetGripperState();

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("state");
  packer.pack(state);

  return response_buf;
}

msgpack::sbuffer handleTerminate() {
  std::cout << "[SERVER] Terminate request received" << std::endl;
  global_shutdown = true;

  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(true);
  packer.pack("error");
  packer.pack(std::string(""));

  return response_buf;
}

msgpack::sbuffer handleUnknownCommand(const std::string &command) {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(false);
  packer.pack("error");
  packer.pack(std::string("Unknown command: ") + command);

  return response_buf;
}

msgpack::sbuffer handleError(const std::string &error_msg) {
  msgpack::sbuffer response_buf;
  msgpack::packer<msgpack::sbuffer> packer(response_buf);

  packer.pack_map(2);
  packer.pack("success");
  packer.pack(false);
  packer.pack("error");
  packer.pack(error_msg);

  return response_buf;
}

void RunServer(const std::string &server_address, franka::Robot *robot,
               franka::Model *model, franka::Gripper *gripper,
               bamboo::controllers::JointImpedanceController *controller,
               bamboo::interpolators::JointInterpolator *interpolator,
               const std::array<double, 7> &joint_lower,
               const std::array<double, 7> &joint_upper) {

  BambooControlServer server(robot, model, gripper, controller, interpolator,
                             joint_lower, joint_upper);

  // Create context and socket
  zmq::context_t context(1);
  zmq::socket_t socket(context, zmq::socket_type::rep);
  socket.bind(server_address);

  std::cout << "Server listening on " << server_address << std::endl;

  // Message handling loop
  while (!global_shutdown) {
    try {
      // Receive message
      zmq::message_t request_msg;
#if CPPZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 3, 0)
      auto result = socket.recv(request_msg, zmq::recv_flags::none);
#else
      auto result = socket.recv(&request_msg, 0);
#endif

      if (!result) {
        continue;
      }

      // Unpack the request
      msgpack::object_handle oh = msgpack::unpack(
          static_cast<const char *>(request_msg.data()), request_msg.size());
      msgpack::object obj = oh.get();

      // Parse as a map to get the command
      std::map<std::string, msgpack::object> request_map;
      obj.convert(request_map);

      std::string command = parseCommand(request_map);
      std::cout << "[SERVER] Received command: " << command << std::endl;

      // Handle the command
      msgpack::sbuffer response_buf;

      try {
        if (command == "get_capabilities") {
          response_buf = handleGetCapabilities();
        } else if (command == "get_robot_state") {
          response_buf = handleGetRobotState(server);
        } else if (command == "start_stream") {
          response_buf = handleStartStream(server, request_map);
        } else if (command == "stream_joint_velocity") {
          response_buf = handleStreamJointVelocity(server, request_map);
        } else if (command == "stop_stream") {
          response_buf = handleStopStream(server);
        } else if (command == "execute_trajectory") {
          response_buf = handleExecuteTrajectory(server, request_map);
        } else if (command == "open_gripper") {
          response_buf = handleOpenGripper(server, request_map);
        } else if (command == "close_gripper") {
          response_buf = handleCloseGripper(server, request_map);
        } else if (command == "get_gripper_state") {
          response_buf = handleGetGripperState(server);
        } else if (command == "terminate") {
          response_buf = handleTerminate();
        } else {
          response_buf = handleUnknownCommand(command);
        }
      } catch (const std::exception &e) {
        std::cerr << "[SERVER] Error handling command: " << e.what()
                  << std::endl;
        response_buf = handleError(e.what());
      }

      // Send response
      zmq::message_t response_msg(response_buf.data(), response_buf.size());
#if CPPZMQ_VERSION >= ZMQ_MAKE_VERSION(4, 3, 0)
      socket.send(response_msg, zmq::send_flags::none);
#else
      socket.send(response_msg,
                  0); // Old API uses reference, not pointer
                      // (https://github.com/zeromq/cppzmq/issues/69)
#endif

    } catch (const zmq::error_t &e) {
      if (e.num() == EINTR || global_shutdown) {
        break;
      }
      std::cerr << "[SERVER] Error: " << e.what() << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "[SERVER] Exception in message loop: " << e.what()
                << std::endl;
    }
  }

  std::cout << "Shutting down server..." << std::endl;
  socket.close();
}

int main(int argc, char **argv) {
  // Register signal handler for graceful shutdown
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::signal(SIGHUP, signalHandler);

  std::string robot_ip;
  std::string port;
  std::string listen_address = "*";  // default
  std::string gripper_type = "none"; // default: no gripper control in C++ node
  std::string robot_model = "fr3";
  bool use_min_jerk = false;

  int opt;
  while ((opt = getopt(argc, argv, "r:p:l:g:e:mh")) != -1) {
    switch (opt) {
    case 'r':
      robot_ip = optarg;
      break;
    case 'p':
      port = optarg;
      break;
    case 'l':
      listen_address = optarg;
      break;
    case 'g':
      gripper_type = optarg;
      break;
    case 'e':
      robot_model = optarg;
      break;
    case 'm':
      use_min_jerk = true;
      break;
    case 'h':
    case '?':
    default:
      std::cerr << "Usage: " << argv[0]
                << " -r <robot-ip> -p <port> [-l <listen-address>] [-g "
                   "<gripper-type>] [-e <fr3|panda>] [-m]"
                << std::endl;
      std::cerr << "  -r: Robot IP address (required)" << std::endl;
      std::cerr << "  -p: Port number (required)" << std::endl;
      std::cerr << "  -l: Listen address (default: * for all interfaces)"
                << std::endl;
      std::cerr << "  -g: Gripper type: 'franka' or 'none' (default: none)"
                << std::endl;
      std::cerr << "  -e: Robot model: 'fr3' or 'panda' (default: fr3)"
                << std::endl;
      std::cerr << "  -m: Use min-jerk interpolation (default: linear)"
                << std::endl;
      std::cerr << "  -h: Show this help" << std::endl;
      return -1;
    }
  }

  // Validate required arguments
  if (robot_ip.empty() || port.empty()) {
    std::cerr << "Error: Robot IP and port are required" << std::endl;
    std::cerr << "Usage: " << argv[0]
              << " -r <robot-ip> -p <port> [-l <listen-address>] [-g "
                 "<gripper-type>] [-e <fr3|panda>] [-m]"
              << std::endl;
    return -1;
  }

  // Validate gripper type
  if (gripper_type != "franka" && gripper_type != "none") {
    std::cerr << "Error: Invalid gripper type '" << gripper_type
              << "'. Must be 'franka' or 'none'" << std::endl;
    return -1;
  }
  if (robot_model != "fr3" && robot_model != "panda") {
    std::cerr << "Error: Invalid robot model '" << robot_model
              << "'. Must be 'fr3' or 'panda'" << std::endl;
    return -1;
  }

  const std::string server_address = "tcp://" + listen_address + ":" + port;

  std::cout << "Bamboo Control Node Starting..." << std::endl;
  std::cout << "Robot IP: " << robot_ip << std::endl;
  std::cout << "Port: " << port << std::endl;
  std::cout << "Listen address: " << listen_address << std::endl;
  std::cout << "Gripper type: " << gripper_type << std::endl;
  std::cout << "Robot model: " << robot_model << std::endl;

  try {
    // Connect to robot
    std::cout << "Connecting to robot..." << std::endl;
    franka::Robot robot(robot_ip);
    robot.automaticErrorRecovery();

    // Set collision behavior
    robot.setCollisionBehavior(
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // Load model
    franka::Model model = robot.loadModel();

    // Conditionally connect to gripper based on gripper_type
    franka::Gripper *gripper_ptr = nullptr;
    std::unique_ptr<franka::Gripper> gripper_holder;

    if (gripper_type == "franka") {
      std::cout << "Connecting to Franka Hand gripper..." << std::endl;
      gripper_holder = std::make_unique<franka::Gripper>(robot_ip);
      gripper_ptr = gripper_holder.get();

      // Home the gripper
      std::cout << "Homing gripper..." << std::endl;
      gripper_ptr->homing();
      std::cout << "Franka Hand gripper ready" << std::endl;
    } else {
      std::cout << "No gripper control in C++ node (gripper_type=none)"
                << std::endl;
    }

    // Create controller and interpolator
    bamboo::controllers::JointImpedanceController controller(&model);
    bamboo::interpolators::InterpolatorType interp_type =
        use_min_jerk ? bamboo::interpolators::InterpolatorType::kMinJerk
                     : bamboo::interpolators::InterpolatorType::kLinear;
    bamboo::interpolators::JointInterpolator interpolator(interp_type);

    // Start server
    const auto &joint_lower =
        robot_model == "fr3" ? kFr3JointLower : kPandaJointLower;
    const auto &joint_upper =
        robot_model == "fr3" ? kFr3JointUpper : kPandaJointUpper;
    RunServer(server_address, &robot, &model, gripper_ptr, &controller,
              &interpolator, joint_lower, joint_upper);

    std::cout << "Control node terminated successfully" << std::endl;

  } catch (const franka::Exception &e) {
    std::cerr << "Franka exception: " << e.what() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
