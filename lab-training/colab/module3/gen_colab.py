#!/usr/bin/env python3
"""Generate Colab versions of Module 3 notebooks"""
import json, os

COLAB_DIR = "/sessions/sharp-relaxed-bell/mnt/23-developskilltrain/lab-training/colab/module3"
NOTEBOOK_DIR = "/sessions/sharp-relaxed-bell/mnt/23-developskilltrain/lab-training/module3-audio-classification/notebooks"

def make_install_cell():
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
            "print('环境配置完成！')\n",
            "\n",
            "# 确认GPU\n",
            "import torch\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU:', torch.cuda.get_device_name(0))\n",
            "else:\n",
            "    print('GPU不可用，请在运行时类型中选择T4 GPU')\n",
            "\n",
            "# 下载ESC-50数据集（如果尚未下载）\n",
            "import os\n",
            "if not os.path.exists('ESC-50'):\n",
            "    print('正在下载ESC-50数据集（约60MB）...')\n",
            "    !wget -q https://github.com/karoldvl/ESC-50/archive/master.zip -O esc50.zip\n",
            "    !unzip -q esc50.zip\n",
            "    !mv ESC-50-master ESC-50\n",
            "    !rm esc50.zip\n",
            "    print('ESC-50下载完成！')\n",
            "else:\n",
            "    print('ESC-50已存在')\n"
        ],
        "execution_count": None,
        "outputs": [],
    }

def convert_to_colab(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    install_cell = make_install_cell()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (Colab)", "language": "python", "name": "python3"}
    nb["metadata"]["accelerator"] = "GPU"
    # Replace local data path with Colab path
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = [s.replace("'../data/ESC-50'", "'ESC-50'").replace('"../data/ESC-50"', '"ESC-50"') for s in cell["source"]]
            # Also replace speech_noise_dataset and intelligibility_dataset paths
            cell["source"] = [s.replace("'../data/speech_noise_dataset'", "'speech_noise_dataset'").replace('"../data/speech_noise_dataset"', '"speech_noise_dataset"') for s in cell["source"]]
            cell["source"] = [s.replace("'../data/intelligibility_dataset'", "'intelligibility_dataset'").replace('"../data/intelligibility_dataset"', '"intelligibility_dataset"') for s in cell["source"]]
    nb["cells"] = [install_cell] + nb["cells"]
    return nb

os.makedirs(COLAB_DIR, exist_ok=True)

for name in ["01-audio-features.ipynb", "02-crnn-classifier.ipynb", "03-ci-tasks.ipynb"]:
    src = os.path.join(NOTEBOOK_DIR, name)
    if not os.path.exists(src):
        print(f"Skip {name} (source not found)")
        continue
    nb = convert_to_colab(src)
    dst = os.path.join(COLAB_DIR, name)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Generated: {dst}")
