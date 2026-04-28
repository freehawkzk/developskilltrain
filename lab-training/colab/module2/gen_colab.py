#!/usr/bin/env python3
"""生成模块2的Colab版本notebook（在原版基础上添加环境安装cell）"""
import json
import os

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(COLAB_DIR)),
                            "module2-dl-intro", "notebooks")

def make_install_cell():
    """生成Colab环境安装cell"""
    return {
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "source": [
            "# ===== Colab环境配置 =====\n",
            "# 运行此cell安装所有依赖（约2-3分钟）\n",
            "!pip install -q numpy scipy matplotlib scikit-learn tqdm\n",
            "!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install -q librosa soundfile tensorboard\n",
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
    """将服务器版notebook转换为Colab版"""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 在第一个cell之前插入安装cell
    install_cell = make_install_cell()

    # 修改kernel spec
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (Colab)",
        "language": "python",
        "name": "python3"
    }
    nb["metadata"]["accelerator"] = "GPU" if any(
        "torch" in str(cell.get("source", ""))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    ) else "NONE"

    # 插入安装cell
    nb["cells"] = [install_cell] + nb["cells"]

    return nb

def main():
    notebook_names = [
        "01-linear-regression.ipynb",
        "02-mlp-pytorch.ipynb",
        "03-cnn-audio.ipynb",
        "04-training-tricks.ipynb",
    ]

    for name in notebook_names:
        src = os.path.join(NOTEBOOK_DIR, name)
        if not os.path.exists(src):
            print(f"跳过 {name}（源文件不存在）")
            continue

        nb = convert_to_colab(src)
        dst = os.path.join(COLAB_DIR, name)
        with open(dst, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"已生成: {dst}")

if __name__ == "__main__":
    main()
