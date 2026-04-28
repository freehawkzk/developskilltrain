#!/usr/bin/env python3
"""Generate Colab versions of Module 4 notebooks"""
import json, os

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(os.path.dirname(COLAB_DIR), "..", "module4-deepace", "notebooks")

def make_install_cell():
    return {
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "source": [
            "# ===== Colab环境配置 =====\n",
            "# 运行此cell安装所有依赖（约2-3分钟）\n",
            "!pip install -q numpy scipy matplotlib pyyaml\n",
            "!pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
            "!pip install -q soundfile librosa\n",
            "print('环境配置完成！')\n",
            "\n",
            "# 确认GPU\n",
            "import torch\n",
            "if torch.cuda.is_available():\n",
            "    print('GPU:', torch.cuda.get_device_name(0))\n",
            "else:\n",
            "    print('GPU不可用，请在运行时类型中选择T4 GPU')\n",
            "\n",
            "# 下载DeepACE代码\n",
            "import os\n",
            "if not os.path.exists('DeepACE_torch'):\n",
            "    print('请上传DeepACE_torch目录到Colab，或从Google Drive挂载')\n",
            "\n",
            "# 生成mini数据集\n",
            "if not os.path.exists('DeepACE_torch/data'):\n",
            "    print('运行prepare_mini_dataset.py生成数据集...')\n",
            "    print('注意：需要先上传代码和数据，或使用合成数据')\n"
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
                line = line.replace("'../ACE'", "'ACE'")
                line = line.replace('"../ACE"', '"ACE"')
                line = line.replace("'../DeepACE_torch'", "'DeepACE_torch'")
                line = line.replace('"../DeepACE_torch"', '"DeepACE_torch"')
                line = line.replace("os.path.join('..', 'ACE'", "os.path.join('ACE'")
                line = line.replace("os.path.join('..', 'DeepACE_torch'", "os.path.join('DeepACE_torch'")
                line = line.replace("os.path.join('..', 'pretrained'", "os.path.join('pretrained'")
                new_source.append(line)
            cell["source"] = new_source
    nb["cells"] = [install_cell] + nb["cells"]
    return nb

for name in ["01-ace-and-paper.ipynb", "02-code-analysis.ipynb", "03-modification-experiments.ipynb"]:
    src = os.path.join(NOTEBOOK_DIR, name)
    if not os.path.exists(src):
        print("Skip %s (source not found)" % name)
        continue
    nb = convert_to_colab(src)
    dst = os.path.join(COLAB_DIR, name)
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Generated: %s" % dst)
