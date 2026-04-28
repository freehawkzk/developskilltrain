#!/usr/bin/env python3
"""生成模块0的Colab版本notebook（在原版基础上添加环境安装cell）"""
import json
import os

COLAB_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(COLAB_DIR)),
                            "module0-python-basics", "notebooks")

def make_install_cell():
    return {
        "cell_type": "code",
        "metadata": {"id": "install_deps"},
        "source": [
            "# ===== Colab环境配置 =====\n",
            "# 运行此cell安装所有依赖（约1-2分钟）\n",
            "!pip install -q numpy scipy matplotlib soundfile torch torchaudio\n",
            "!apt-get install -qq ffmpeg sox\n",
            "print('✅ 环境配置完成！')"
        ],
        "execution_count": None,
        "outputs": [],
    }

def add_colab_metadata(cell):
    """为cell添加Colab兼容的metadata"""
    new_cell = dict(cell)
    if "id" not in new_cell.get("metadata", {}):
        if "id" not in new_cell["metadata"]:
            new_cell["metadata"] = dict(new_cell.get("metadata", {}))
    return new_cell

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
        "01-basics-signal.ipynb",
        "02-oop-audio.ipynb",
        "03-debugging-git.ipynb",
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
