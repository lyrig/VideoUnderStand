#!/bin/bash
# 文件：~/run.sh
# 用法：srun -p raise --gres=gpu:8 --quotatype=reserved bash ~/run.sh python train.py

# === 用户配置区 (请修改此处) ===
# 镜像文件的绝对路径
CONTAINER="/mnt/petrelfs/panjiabao/huxiaobin/shy/ubuntu_22.04.sif"
# Conda 初始化脚本路径
CONDA_SH="~/miniconda3/etc/profile.d/conda.sh"
# 目标 Conda 环境名称
ENV_NAME="main"
# ==============================

# 执行容器指令
# --nv: 启用 NVIDIA GPU 支持
# -B /mnt:/mnt: 将宿主机的 /mnt 挂载到容器内，确保能访问代码和数据
apptainer exec --nv -B /mnt:/mnt ${CONTAINER} \
    bash -c "source ${CONDA_SH} && conda activate ${ENV_NAME} && $*"