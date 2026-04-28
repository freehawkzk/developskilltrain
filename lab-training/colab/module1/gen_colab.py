#!/usr/bin/env python3
"""生成模块1的Colab版本notebook"""
import json
import os

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(COLAB_DIR)),
                            "module1-linux-env", "notebooks")

def make_install_cell():
    return {
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "source": [
            "# ===== Colab环境配置 =====\n",
            "# 运行此cell安装所有依赖（约2-3分钟）\n",
            "!pip install -q numpy scipy matplotlib scikit-learn tqdm\n",
            "!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install -q librosa soundfile pesq pystoi\n",
            "!apt-get install -qq ffmpeg sox\n",
            "print('✅ 环境配置完成！')\n",
            "\n",
            "# 确认GPU\n",
            "import torch\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
            "else:\n",
            "    print('⚠️ GPU不可用，请在运行时类型中选择T4 GPU')"
        ],
        "execution_count": None,
        "outputs": [],
    }

def convert_to_colab(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    install_cell = make_install_cell()

    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (Colab)",
        "language": "python",
        "name": "python3"
    }

    # Module 1 Session 1 is about Linux commands, not really suited for Colab
    # We only convert Session 2 (environment setup)
    nb["cells"] = [install_cell] + nb["cells"]

    return nb

def main():
    # Only convert notebook 2 (environment setup) for Colab
    # Notebook 1 (Linux survival) requires a real terminal
    src = os.path.join(NOTEBOOK_DIR, "02-environment-setup.ipynb")
    if os.path.exists(src):
        nb = convert_to_colab(src)
        dst = os.path.join(COLAB_DIR, "02-environment-setup.ipynb")
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"已生成: {dst}")

    # Create a note about notebook 1
    note = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (Colab)",
                "language": "python",
                "name": "python3"
            }
        },
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Linux生存指南\n",
                "\n",
                "⚠️ **本课内容需要在真实的Linux终端中操作，不适合在Colab中运行。**\n",
                "\n",
                "请在以下环境中完成本课练习：\n",
                "- SSH连接到租用的服务器\n",
                "- 本地安装的Linux虚拟机（如WSL2、VirtualBox）\n",
                "- 本地Mac/Linux终端\n",
                "\n",
                "命令速查表请参考：[Linux速查表](../module1-linux-env/cheatsheet-linux.md)\n"
            ]
        }]
    }
    note_path = os.path.join(COLAB_DIR, "01-linux-survival.ipynb")
    with open(note_path, 'w', encoding='utf-8') as f:
        json.dump(note, f, ensure_ascii=False, indent=1)
    print(f"已生成: {note_path}")

if __name__ == "__main__":
    main()
