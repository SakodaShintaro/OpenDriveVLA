#!/usr/bin/env python

"""
OpenDriveVLA Agent for CARLA Leaderboard 2.1
"""

import os
import re

import carla
import numpy as np
import torch

# Leaderboardのベースクラスをインポート
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token

# OpenDriveVLAのモデルとユーティリティをインポート
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


class OpenDriveVLAAgent(AutonomousAgent):
    """
    OpenDriveVLA Agent for CARLA Leaderboard
    """

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        self.track = Track.SENSORS
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.warned_model_not_loaded = False

    def setup(self, path_to_conf_file):
        """
        モデルとセンサーの初期化

        Args:
            path_to_conf_file: 設定ファイルへのパス
        """
        # モデルのロード
        model_path = os.environ.get(
            "DRIVEVLA_MODEL_PATH",
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--OpenDriveVLA--OpenDriveVLA-0.5B/snapshots/5b219caae79ae65b7068fc9cf442220b75f07dff"
            ),
        )

        print(f"Loading OpenDriveVLA model from: {model_path}")

        disable_torch_init()

        # モデルのロード
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": "flash_attention_2",
        }

        overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
        llava_model_args["overwrite_config"] = overwrite_config

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base=None,
            model_name="llava_qwen",
            device_map=self.device,
            **llava_model_args,
        )

        self.model.eval()
        self.model_loaded = True

        print("OpenDriveVLA model loaded successfully!")

    def sensors(self):
        """
        センサー設定を定義

        OpenDriveVLAが必要とするセンサー:
        - 前方カメラ（RGB）
        - 複数視点のカメラ（前方、左前、右前、左後、右後、後方）
        - GPS
        - IMU
        - 速度センサー
        """
        sensors = [
            # 前方カメラ（メイン）
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_FRONT",
            },
            # 前方左カメラ
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": -0.5,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -55.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_FRONT_LEFT",
            },
            # 前方右カメラ
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.5,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 55.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_FRONT_RIGHT",
            },
            # 後方左カメラ
            {
                "type": "sensor.camera.rgb",
                "x": -1.0,
                "y": -0.5,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -110.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_BACK_LEFT",
            },
            # 後方右カメラ
            {
                "type": "sensor.camera.rgb",
                "x": -1.0,
                "y": 0.5,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 110.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_BACK_RIGHT",
            },
            # 後方カメラ
            {
                "type": "sensor.camera.rgb",
                "x": -1.0,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 180.0,
                "width": 1600,
                "height": 900,
                "fov": 110,
                "id": "CAM_BACK",
            },
            # GPS
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "id": "GPS"},
            # IMU
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0, "id": "IMU"},
            # 速度センサー
            {"type": "sensor.speedometer", "id": "SPEED"},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        1ステップの推論を実行

        Args:
            input_data: センサーデータの辞書
            timestamp: タイムスタンプ

        Returns:
            carla.VehicleControl: 車両制御コマンド
        """
        # デフォルトの制御
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.3
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        # モデルが未ロードの場合は最初の1回だけ警告を出す
        if not self.model_loaded:
            if not self.warned_model_not_loaded:
                print("[Warning] Model not loaded, using default control")
                self.warned_model_not_loaded = True
            return control

        # 1. センサーデータから画像を取得
        images = self._get_camera_images(input_data)
        if not images or "CAM_FRONT" not in images:
            print("[Warning] No camera images available, using default control")
            return control

        # 2. 車両状態を取得
        vehicle_state = self._get_vehicle_state(input_data)

        # 3. プロンプトを生成
        prompt = self._generate_prompt(vehicle_state)

        # 4. モデルで推論
        prediction = self._run_inference(images, prompt)
        if prediction is None:
            print("[Warning] Inference returned None, using default control")
            return control

        # 5. 予測結果から制御コマンドを生成
        control = self._prediction_to_control(prediction, vehicle_state)

        return control

    def _get_camera_images(self, input_data):
        """センサーデータから画像を取得"""
        images = {}
        camera_ids = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]

        for cam_id in camera_ids:
            if cam_id not in input_data:
                continue

            sensor_data = input_data[cam_id][1]
            if sensor_data is None or not hasattr(sensor_data, "shape"):
                print(f"[Warning] Invalid sensor data for {cam_id}")
                continue

            # CARLAのセンサーデータからPIL Imageに変換
            # BGRAからRGBに変換
            array = np.frombuffer(sensor_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (sensor_data.shape[0], sensor_data.shape[1], 4))
            array = array[:, :, :3]  # BGRAのうちRGBだけ取得
            array = array[:, :, ::-1]  # BGRからRGBに変換
            images[cam_id] = Image.fromarray(array)

        return images

    def _get_vehicle_state(self, input_data):
        """車両状態を取得"""
        state = {
            "speed": 0.0,
            "acceleration": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }

        # 速度
        if "SPEED" in input_data:
            speed_data = input_data["SPEED"][1]
            state["speed"] = speed_data["speed"] if isinstance(speed_data, dict) else 0.0

        # IMU（加速度・角速度）
        if "IMU" in input_data:
            imu_data = input_data["IMU"][1]
            if hasattr(imu_data, "accelerometer"):
                acc = imu_data.accelerometer
                state["acceleration"] = [acc.x, acc.y, acc.z]
            if hasattr(imu_data, "gyroscope"):
                gyro = imu_data.gyroscope
                state["angular_velocity"] = [gyro.x, gyro.y, gyro.z]

        return state

    def _generate_prompt(self, vehicle_state):
        """車両状態からプロンプトを生成"""
        speed = vehicle_state["speed"]
        acc = vehicle_state["acceleration"]
        gyro = vehicle_state["angular_velocity"]

        # OpenDriveVLA風のプロンプト
        prompt = (
            f"<image>\n"
            f"Given the current driving scenario with the following ego vehicle states:\n"
            f"- Speed: {speed:.2f} m/s\n"
            f"- Acceleration: ({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}) m/s²\n"
            f"- Angular Velocity: ({gyro[0]:.2f}, {gyro[1]:.2f}, {gyro[2]:.2f}) rad/s\n"
            f"Please plan the future trajectory and provide driving actions.\n"
            f"Output format: steering: <value>, throttle: <value>, brake: <value>"
        )

        return prompt

    def _run_inference(self, images, prompt):
        """モデルで推論を実行"""
        # メイン画像として前方カメラを使用
        if "CAM_FRONT" not in images:
            return None

        image = images["CAM_FRONT"]

        # 画像を前処理
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # プロンプトをトークナイズ
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )

        # 推論実行
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=512,
                use_cache=True,
            )

        # 出力をデコード
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return outputs

    def _prediction_to_control(self, prediction, vehicle_state):
        """予測結果から制御コマンドを生成"""
        control = carla.VehicleControl()

        # デフォルト値
        steer = 0.0
        throttle = 0.3
        brake = 0.0

        if prediction is None:
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            control.hand_brake = False
            control.manual_gear_shift = False
            return control

        # テキスト出力から制御値を抽出
        # パターンマッチングで数値を抽出
        steering_match = re.search(r"steering[:\s]+([+-]?[\d.]+)", prediction, re.IGNORECASE)
        throttle_match = re.search(r"throttle[:\s]+([+-]?[\d.]+)", prediction, re.IGNORECASE)
        brake_match = re.search(r"brake[:\s]+([+-]?[\d.]+)", prediction, re.IGNORECASE)

        # マッチした値を float に変換（失敗時はデフォルト値を使用）
        if (
            steering_match
            and steering_match.group(1).replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).isdigit()
        ):
            steer = float(steering_match.group(1))
        if (
            throttle_match
            and throttle_match.group(1).replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).isdigit()
        ):
            throttle = float(throttle_match.group(1))
        if brake_match and brake_match.group(1).replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).isdigit():
            brake = float(brake_match.group(1))

        # 値の範囲を制限
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def destroy(self):
        """
        クリーンアップ
        """
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        print("OpenDriveVLA agent destroyed")
