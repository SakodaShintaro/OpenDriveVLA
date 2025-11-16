#!/bin/bash

# OpenDriveVLA Leaderboard 2.1 評価スクリプト

# 環境変数の設定
export CARLA_ROOT=${CARLA_ROOT:-~/CARLA_0.9.16}
export SCENARIO_RUNNER_ROOT=${SCENARIO_RUNNER_ROOT:-${CARLA_ROOT}/scenario_runner}
export LEADERBOARD_ROOT=${LEADERBOARD_ROOT:-${CARLA_ROOT}/leaderboard}
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"

# OpenDriveVLAモデルのパス
export DRIVEVLA_MODEL_PATH=${DRIVEVLA_MODEL_PATH:-~/.cache/huggingface/hub/models--OpenDriveVLA--OpenDriveVLA-0.5B/snapshots/5b219caae79ae65b7068fc9cf442220b75f07dff}

# CARLAサーバーの設定
export CARLA_HOST=${CARLA_HOST:-localhost}
export CARLA_PORT=${CARLA_PORT:-2000}
export CARLA_TRAFFIC_MANAGER_PORT=${CARLA_TRAFFIC_MANAGER_PORT:-8000}

# 評価結果の出力ディレクトリ
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="output/leaderboard_results/${TIMESTAMP}"
mkdir -p ${RESULTS_DIR}

echo "========================================="
echo "OpenDriveVLA Leaderboard 2.1 Evaluation"
echo "========================================="
echo "CARLA_ROOT: ${CARLA_ROOT}"
echo "SCENARIO_RUNNER_ROOT: ${SCENARIO_RUNNER_ROOT}"
echo "LEADERBOARD_ROOT: ${LEADERBOARD_ROOT}"
echo "DRIVEVLA_MODEL_PATH: ${DRIVEVLA_MODEL_PATH}"
echo "CARLA_HOST: ${CARLA_HOST}"
echo "CARLA_PORT: ${CARLA_PORT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "========================================="

# ルート設定ファイル（デフォルトのテストルートを使用）
ROUTES=${ROUTES:-${LEADERBOARD_ROOT}/data/routes_training.xml}
SCENARIOS=${SCENARIOS:-${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json}

# エージェントの設定
AGENT_CONFIG="drivevla/leaderboard_agent.py"
CHECKPOINT_ENDPOINT="${RESULTS_DIR}/checkpoint.json"
RESUME=${RESUME:-0}

# Leaderboardの実行
python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes=${ROUTES} \
    --scenarios=${SCENARIOS} \
    --agent=${AGENT_CONFIG} \
    --agent-config="" \
    --track=SENSORS \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --timeout=600.0 \
    --traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT} \
    --resume=${RESUME} \
    2>&1 | tee ${RESULTS_DIR}/evaluation.log

echo "========================================="
echo "Evaluation completed!"
echo "Results saved to: ${RESULTS_DIR}"
echo "========================================="
