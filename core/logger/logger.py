import csv
import os
import torch
import time


class Logger:
    def __init__(self, scene, log_dir="data_logs"):
        """ Initializes logger and creates necessary directories/files. """
        self.scene = scene
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Define log file names
        self.files = {
            "base_pos_and_ori": os.path.join(log_dir, "base_pos_and_ori_.csv"),
            "joint_efforts": os.path.join(log_dir, "joint_efforts.csv"),
            "joint_velocities": os.path.join(log_dir, "joint_velocities.csv"),
            "joint_positions": os.path.join(log_dir, "joint_positions.csv"),
        }

        # Initialize files with headers
        self._init_csv_files()

    def _init_csv_files(self):
        """ Initializes CSV files with appropriate headers. """
        headers = {

            "base_pos_and_ori": ["Time(ms)", "Base_X", "Base_Y", "Base_Z", "Roll", "Pitch", "Yaw"],

            "joint_efforts": ["Time(ms)", "FR_HAA",
                              "FR_HFE",
                              "FR_KFE",
                              "FL_HAA",
                              "FL_HFE",
                              "FL_KFE",
                              "HR_HAA",
                              "HR_HFE",
                              "HR_KFE",
                              "HL_HAA",
                              "HL_HFE",
                              "HL_KFE", ],
            "joint_velocities": ["Time(ms)", "FR_HAA",
                                 "FR_HFE",
                                 "FR_KFE",
                                 "FL_HAA",
                                 "FL_HFE",
                                 "FL_KFE",
                                 "HR_HAA",
                                 "HR_HFE",
                                 "HR_KFE",
                                 "HL_HAA",
                                 "HL_HFE",
                                 "HL_KFE", ],
            "joint_positions": ["Time(ms)", "FR_HAA",
                                "FR_HFE",
                                "FR_KFE",
                                "FL_HAA",
                                "FL_HFE",
                                "FL_KFE",
                                "HR_HAA",
                                "HR_HFE",
                                "HR_KFE",
                                "HL_HAA",
                                "HL_HFE",
                                "HL_KFE", ]
        }

        for log_type, file_path in self.files.items():
            if not os.path.exists(file_path):  # Prevent overwriting
                with open(file_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers[log_type])

    def _tensor_to_string(self, tensor):
        return ', '.join(map(str, tensor.tolist()))

    def log_base_pos_and_ori(self, base_pos, bas_quat):
        timestamp = self.scene.cur_t
        roll, pitch, yaw = self._quat_to_euler(bas_quat)
        data = [timestamp] + base_pos.tolist() + [roll, pitch, yaw]
        self._write_to_csv("base_pos_and_ori", data)

    def log_joint_positions(self, joint_positions):
        timestamp = self.scene.cur_t
        data = [timestamp] + joint_positions.tolist()
        self._write_to_csv("joint_positions", data)

    def log_joint_velocities(self, joint_velocites):
        timestamp = self.scene.cur_t
        data = [timestamp] + joint_velocites.tolist()
        self._write_to_csv("joint_velocities", data)

    def log_joint_efforts(self, joint_efforts):
        timestamp = self.scene.cur_t
        data = [timestamp] + joint_efforts.tolist()
        self._write_to_csv("joint_efforts", data)

    def _write_to_csv(self, log_type, data):
        """ Writes data to a CSV file. """
        with open(self.files[log_type], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def _quat_to_euler(self, quat):
        """ Converts quaternion to Euler angles (roll, pitch, yaw). """
        x, y, z, w = quat
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
