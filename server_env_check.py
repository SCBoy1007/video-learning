#!/usr/bin/env python3
"""
服务器环境信息收集脚本
用于Video-Learning项目部署前的环境检查和调试
"""

import sys
import os
import platform
import subprocess
import json
import datetime
import socket
from pathlib import Path


class ServerEnvChecker:
    def __init__(self):
        self.report = {}
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_command(self, cmd, shell=True):
        """安全地运行命令并返回输出"""
        try:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip(), result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return "Command failed or timed out", False

    def check_system_info(self):
        """收集系统基本信息"""
        print("🔍 检查系统信息...")

        self.report['system'] = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'os': platform.system(),
            'os_release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'timestamp': self.timestamp
        }

        # 系统运行时间
        uptime, _ = self.run_command("uptime")
        self.report['system']['uptime'] = uptime

        # 内核版本
        kernel, _ = self.run_command("uname -r")
        self.report['system']['kernel'] = kernel

    def check_hardware_info(self):
        """收集硬件信息"""
        print("🖥️  检查硬件信息...")

        hardware = {}

        # CPU信息
        cpu_info, _ = self.run_command("lscpu | grep -E '(Model name|Core|Thread|CPU MHz)'")
        hardware['cpu'] = cpu_info.split('\n') if cpu_info else ["CPU info not available"]

        # 内存信息
        mem_info, _ = self.run_command("free -h")
        hardware['memory'] = mem_info

        # 磁盘使用情况
        disk_info, _ = self.run_command("df -h")
        hardware['disk_usage'] = disk_info

        self.report['hardware'] = hardware

    def check_gpu_info(self):
        """收集GPU信息"""
        print("🎮 检查GPU信息...")

        gpu = {}

        # NVIDIA GPU信息
        nvidia_smi, success = self.run_command("nvidia-smi")
        if success:
            gpu['nvidia_smi'] = nvidia_smi

            # GPU详细信息
            gpu_details, _ = self.run_command(
                "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv"
            )
            gpu['gpu_details'] = gpu_details
        else:
            gpu['nvidia_smi'] = "NVIDIA GPU not detected or nvidia-smi not available"

        # CUDA版本
        cuda_version, _ = self.run_command("nvcc --version")
        gpu['cuda_version'] = cuda_version if cuda_version else "CUDA not available"

        # NVIDIA驱动版本
        driver_version, _ = self.run_command("cat /proc/driver/nvidia/version")
        gpu['driver_version'] = driver_version if driver_version else "NVIDIA driver info not available"

        self.report['gpu'] = gpu

    def check_python_env(self):
        """检查Python环境"""
        print("🐍 检查Python环境...")

        python_env = {
            'python_executable': sys.executable,
            'python_path': sys.path[:5],  # 只显示前5个路径
            'virtual_env': os.environ.get('VIRTUAL_ENV', 'Not in virtual environment'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
        }

        # Conda信息
        conda_info, success = self.run_command("conda info --envs")
        if success:
            python_env['conda_envs'] = conda_info

        self.report['python_env'] = python_env

    def check_ml_packages(self):
        """检查机器学习相关包版本"""
        print("📦 检查ML包版本...")

        packages_to_check = [
            'torch', 'torchvision', 'transformers', 'datasets', 'accelerate',
            'numpy', 'pandas', 'pillow', 'wandb', 'ray', 'vllm'
        ]

        package_info = {}

        for package in packages_to_check:
            try:
                __import__(package)
                version_cmd = f"python -c \"import {package}; print({package}.__version__)\""
                version, success = self.run_command(version_cmd)
                package_info[package] = version if success else "Version check failed"
            except ImportError:
                package_info[package] = "Not installed"

        self.report['ml_packages'] = package_info

        # pip list 关键包
        pip_list, _ = self.run_command("pip list | grep -E '(torch|transform|dataset|accelerate|wandb|ray|vllm)'")
        self.report['pip_packages'] = pip_list

    def check_network_storage(self):
        """检查网络和存储"""
        print("🌐 检查网络和存储...")

        network_storage = {}

        # 网络接口
        interfaces, _ = self.run_command("ip addr show")
        network_storage['network_interfaces'] = interfaces

        # 互联网连接测试
        ping_test, success = self.run_command("ping -c 1 google.com")
        network_storage['internet_connectivity'] = "Connected" if success else "Connection failed"

        # 检查数据目录（如果存在）
        data_dirs = ['data/', '/data/', '/mnt/data/', 'datasets/']
        existing_dirs = []
        for dir_path in data_dirs:
            if os.path.exists(dir_path):
                size_cmd = f"du -sh {dir_path}"
                size_info, _ = self.run_command(size_cmd)
                existing_dirs.append(f"{dir_path}: {size_info}")

        network_storage['data_directories'] = existing_dirs

        # 挂载点
        mounts, _ = self.run_command("mount | grep -E '(ext4|xfs|nfs)'")
        network_storage['mounts'] = mounts

        self.report['network_storage'] = network_storage

    def check_resources(self):
        """检查当前资源使用"""
        print("📊 检查资源使用情况...")

        resources = {}

        # CPU和内存使用
        top_info, _ = self.run_command("top -bn1 | head -15")
        resources['current_usage'] = top_info

        # GPU使用情况
        if 'nvidia_smi' in self.report.get('gpu', {}):
            gpu_usage, _ = self.run_command("nvidia-smi -q -d UTILIZATION,MEMORY")
            resources['gpu_usage'] = gpu_usage

        # 相关进程
        python_processes, _ = self.run_command("ps aux | grep python | head -10")
        resources['python_processes'] = python_processes

        self.report['resources'] = resources

    def check_project_requirements(self):
        """检查项目特定需求"""
        print("🎯 检查项目特定需求...")

        project = {}

        # 检查当前目录
        current_dir = os.getcwd()
        project['current_directory'] = current_dir
        project['directory_contents'] = os.listdir(current_dir)[:20]  # 前20个文件

        # 检查requirements.txt
        if os.path.exists('requirements.txt'):
            with open('requirements.txt', 'r') as f:
                project['requirements'] = f.read()

        # 检查预训练模型目录
        model_dirs = ['pretrained_models/', 'models/', '/models/']
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                models, _ = self.run_command(f"ls -la {model_dir}")
                project[f'models_in_{model_dir}'] = models

        # 检查CUDA_VISIBLE_DEVICES
        project['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')

        self.report['project'] = project

    def generate_report(self):
        """生成完整报告"""
        print("📋 生成环境报告...")

        # 运行所有检查
        self.check_system_info()
        self.check_hardware_info()
        self.check_gpu_info()
        self.check_python_env()
        self.check_ml_packages()
        self.check_network_storage()
        self.check_resources()
        self.check_project_requirements()

        return self.report

    def save_report(self, filename=None):
        """保存报告到文件"""
        if filename is None:
            hostname = socket.gethostname()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"server_env_report_{hostname}_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        return filename

    def print_summary(self):
        """打印关键信息摘要"""
        print("\n" + "="*80)
        print("🚀 服务器环境检查摘要")
        print("="*80)

        # 系统信息
        sys_info = self.report.get('system', {})
        print(f"🖥️  主机: {sys_info.get('hostname', 'Unknown')}")
        print(f"🐧 系统: {sys_info.get('platform', 'Unknown')}")

        # GPU信息
        gpu_info = self.report.get('gpu', {})
        if 'nvidia-smi not available' not in gpu_info.get('nvidia_smi', ''):
            gpu_details = gpu_info.get('gpu_details', '')
            if gpu_details:
                gpu_lines = gpu_details.split('\n')[1:]  # 跳过标题
                print(f"🎮 GPU信息:")
                for line in gpu_lines:
                    if line.strip():
                        print(f"    {line}")

        # Python和关键包
        python_env = self.report.get('python_env', {})
        print(f"🐍 Python: {python_env.get('python_executable', 'Unknown')}")
        print(f"📦 环境: {python_env.get('conda_env', python_env.get('virtual_env', 'Unknown'))}")

        # 关键包版本
        ml_packages = self.report.get('ml_packages', {})
        key_packages = ['torch', 'transformers', 'datasets', 'wandb']
        print("📦 关键包版本:")
        for pkg in key_packages:
            version = ml_packages.get(pkg, 'Not found')
            print(f"    {pkg}: {version}")

        print("="*80)


def main():
    print("🔥 Video-Learning 服务器环境检查工具")
    print("="*80)

    checker = ServerEnvChecker()
    report = checker.generate_report()

    # 保存报告
    report_file = checker.save_report()
    print(f"\n✅ 完整报告已保存到: {report_file}")

    # 打印摘要
    checker.print_summary()

    print(f"\n💡 建议:")
    print("1. 检查GPU和CUDA是否正确安装")
    print("2. 确认所需Python包版本是否匹配")
    print("3. 验证数据目录和模型目录是否可访问")
    print("4. 检查网络连接是否正常（下载模型需要）")


if __name__ == "__main__":
    main()