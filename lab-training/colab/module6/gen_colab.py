#!/usr/bin/env python3
"""Generate Colab versions of Module 6 notebooks"""
import json, os

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(os.path.dirname(COLAB_DIR), "..", "module6-asr", "notebooks")

def make_install_cell():
    return {
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "source": [
            "# ===== Colab环境配置 =====\n",
            "# 运行此cell安装所有依赖（约3-5分钟）\n",
            "!pip install -q numpy scipy matplotlib\n",
            "!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install -q openai-whisper\n",
            "!pip install -q jieba\n",
            "print('环境配置完成！')\n",
            "\n",
            "# 确认GPU\n",
            "import torch\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU:', torch.cuda.get_device_name(0))\n",
            "else:\n",
            "    print('GPU不可用，请在运行时类型中选择T4 GPU')\n",
            "\n",
            "# 预下载Whisper模型\n",
            "import whisper\n",
            "print('正在下载Whisper small模型...')\n",
            "model = whisper.load_model('small')\n",
            "print('Whisper模型就绪！')\n"
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
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                line = line.replace("'..', 'test_audio'", "'test_audio'")
                line = line.replace("'..', '..', 'module5-deepfilternet', 'test_samples'", "'test_samples'")
                new_source.append(line)
            cell["source"] = new_source
    nb["cells"] = [install_cell] + nb["cells"]
    return nb

for name in ["01-asr-principles.ipynb", "02-whisper-pipeline.ipynb"]:
    src = os.path.join(NOTEBOOK_DIR, name)
    if not os.path.exists(src):
        print("Skip %s (source not found)" % name)
        continue
    nb = convert_to_colab(src)
    dst = os.path.join(COLAB_DIR, name)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Generated: %s" % dst)
