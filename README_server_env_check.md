# 服务器环境检查工具

## 概述

`server_env_check.py` 是一个全面的服务器环境信息收集脚本，专为 Video-Learning 项目部署前的环境检查和调试而设计。

## 功能特性

### 🔍 收集的信息类别

1. **系统信息**
   - 主机名、操作系统、内核版本
   - Python版本和路径
   - 系统运行时间

2. **硬件信息**
   - CPU详细信息（型号、核心数等）
   - 内存使用情况
   - 磁盘使用状况

3. **GPU信息**
   - NVIDIA GPU详细信息（型号、显存）
   - CUDA版本和驱动版本
   - GPU实时使用率

4. **Python环境**
   - 虚拟环境/Conda环境信息
   - Python路径配置
   - 环境列表

5. **机器学习包版本**
   - PyTorch、Transformers、Datasets等关键包
   - 版本兼容性检查
   - 缺失包识别

6. **网络和存储**
   - 网络接口配置
   - 互联网连接测试
   - 数据目录检查
   - 挂载点信息

7. **资源使用情况**
   - 当前CPU、内存、GPU使用率
   - 运行中的Python进程
   - 系统负载

8. **项目特定检查**
   - 当前工作目录
   - 预训练模型目录
   - CUDA_VISIBLE_DEVICES设置
   - requirements.txt检查

## 使用方法

### 在服务器上运行

```bash
# 1. 确保在Video-Learning项目根目录
cd /path/to/video-learning

# 2. 运行环境检查
python server_env_check.py
```

### 输出结果

1. **控制台输出**: 实时显示检查进度和关键信息摘要
2. **JSON报告**: 详细的机器可读环境信息
   - 文件名格式: `server_env_report_{hostname}_{timestamp}.json`

## 输出示例

```
🔥 Video-Learning 服务器环境检查工具
================================================================================
🚀 服务器环境检查摘要
================================================================================
🖥️  主机: gpu-server-01
🐧 系统: Linux-5.4.0-84-generic-x86_64-with-ubuntu-20.04
🎮 GPU信息:
    Tesla A800-SXM4-80GB, 81251 MiB, 1024 MiB, 80227 MiB, 0 %
    Tesla A800-SXM4-80GB, 81251 MiB, 1024 MiB, 80227 MiB, 0 %
🐍 Python: /opt/conda/envs/visionreasoner/bin/python
📦 环境: visionreasoner
📦 关键包版本:
    torch: 2.6.0
    transformers: 4.45.2
    datasets: 3.1.0
    wandb: 0.18.5
================================================================================
```

## 故障排除建议

脚本会自动检测并提供以下建议：

1. **GPU检查**: 验证NVIDIA驱动和CUDA安装
2. **包版本**: 确认依赖包版本兼容性
3. **存储**: 检查数据目录和模型目录可访问性
4. **网络**: 验证模型下载所需的网络连接

## 注意事项

- 脚本设计为在Linux服务器上运行，但也兼容macOS（用于本地开发测试）
- 某些系统命令可能需要适当的权限
- 网络检查包含对外部服务的ping测试
- GPU检查依赖于nvidia-smi工具的可用性

## 在训练前的典型使用流程

```bash
# 1. 拉取最新代码
git pull origin master

# 2. 运行环境检查
python server_env_check.py

# 3. 根据报告调整配置（如有需要）

# 4. 开始训练
bash training_scripts/run_brain_tumor_3d_4x80G.sh
```

此工具帮助在训练开始前识别潜在的环境问题，避免训练过程中的意外中断。