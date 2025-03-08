import csv
import os
import torch
import time


class Logger:
    def __init__(self, log_dir="data_logs"):
        """ Initializes logger and creates necessary directories/files. """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Define log file names
        self.files = {
            "state_observation": os.path.join(log_dir, "state_observation_log.csv"),
            "action_patterns": os.path.join(log_dir, "action_patterns_log.csv"),
            "rewards": os.path.join(log_dir, "rewards_log.csv"),
            "dynamics": os.path.join(log_dir, "dynamics_log.csv"),
            "implementation": os.path.join(log_dir, "implementation_log.csv"),
        }

        # Initialize files with headers
        self._init_csv_files()

        # Timer start
        self.start_time = time.time()

    def _init_csv_files(self):
        """ Initializes CSV files with appropriate headers. """
        headers = {
            "state_observation": ["Time(ms)", "Base_X", "Base_Y", "Base_Z", "Roll", "Pitch", "Yaw", "Lin_Vel_X",
                                  "Lin_Vel_Y", "Lin_Vel_Z", "Ang_Vel_X", "Ang_Vel_Y", "Ang_Vel_Z"],
            "action_patterns": ["Time(ms)", "Commanded_Action_1", "Executed_Action_1", "Action_Change_Rate_1", "..."],
            "rewards": ["Time(ms)", "Tracking_Lin_Vel", "Tracking_Ang_Vel", "Lin_Vel_Z", "Action_Rate", "Base_Height"],
            "dynamics": ["Time(ms)", "Contact_Force_X", "Contact_Force_Y", "Contact_Force_Z", "Joint_Torque_1", "..."],
            "implementation": ["Time(ms)", "PD_Target_1", "PD_Actual_1", "Action_Scaling_Before",
                               "Action_Scaling_After", "..."]
        }

        for log_type, file_path in self.files.items():
            if not os.path.exists(file_path):  # Prevent overwriting
                with open(file_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers[log_type])

    def log_state_observation(self, base_pos, base_quat, lin_vel, ang_vel):
        """ Logs base state, orientation, velocities. """
        timestamp = int((time.time() - self.start_time) * 1000)
        roll, pitch, yaw = self._quat_to_euler(base_quat)
        data = [timestamp] + base_pos.tolist() + [roll, pitch, yaw] + lin_vel.tolist() + ang_vel.tolist()
        self._write_to_csv("state_observation", data)

    def log_action_patterns(self, commanded_action, executed_action, action_rate):
        """ Logs action patterns. """
        timestamp = int((time.time() - self.start_time) * 1000)
        data = [timestamp] + commanded_action.tolist() + executed_action.tolist() + action_rate.tolist()
        self._write_to_csv("action_patterns", data)

    def log_rewards(self, reward_dict):
        """ Logs reward components separately. """
        timestamp = int((time.time() - self.start_time) * 1000)
        data = [timestamp] + [reward_dict.get(k, 0) for k in
                              ["tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "action_rate", "base_height"]]
        self._write_to_csv("rewards", data)

    def log_dynamics(self, contact_forces, joint_torques):
        """ Logs environmental interactions like contact forces and joint torques. """
        timestamp = int((time.time() - self.start_time) * 1000)
        data = [timestamp] + contact_forces.tolist() + joint_torques.tolist()
        self._write_to_csv("dynamics", data)

    def log_implementation(self, pd_target, pd_actual, action_scaling_before, action_scaling_after):
        """ Logs PD control performance and action scaling effects. """
        timestamp = int((time.time() - self.start_time) * 1000)
        data = [timestamp] + pd_target.tolist() + pd_actual.tolist() + [action_scaling_before, action_scaling_after]
        self._write_to_csv("implementation", data)

    def _write_to_csv(self, log_type, data):
        """ Writes data to a CSV file. """
        with open(self.files[log_type], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def _quat_to_euler(self, quat):
        """ Converts quaternion to Euler angles (roll, pitch, yaw). """
        x, y, z, w = quat[0]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, +1.0)
        pitch = torch.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)

        return roll.item(), pitch.item(), yaw.item()
