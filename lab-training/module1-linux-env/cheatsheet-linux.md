# Linux速查表

## 文件与目录操作

| 命令 | 功能 | 示例 |
|------|------|------|
| `pwd` | 显示当前目录 | `pwd` |
| `ls` | 列出目录内容 | `ls -la` |
| `cd` | 切换目录 | `cd /home/user/data` |
| `mkdir` | 创建目录 | `mkdir -p project/data` |
| `cp` | 复制文件 | `cp file.txt backup/` |
| `mv` | 移动/重命名 | `mv old.txt new.txt` |
| `rm` | 删除文件 | `rm file.txt` |
| `rm -r` | 删除目录 | `rm -r old_dir/` |
| `cat` | 查看文件内容 | `cat config.yaml` |
| `less` | 分页查看 | `less log.txt` |
| `head` | 查看文件开头 | `head -20 data.csv` |
| `tail` | 查看文件末尾 | `tail -f training.log` |
| `touch` | 创建空文件 | `touch new_file.py` |
| `wc` | 统计行数/字数 | `wc -l data.csv` |
| `du` | 查看目录大小 | `du -sh data/` |
| `df` | 查看磁盘空间 | `df -h` |
| `which` | 查看命令位置 | `which python` |
| `find` | 查找文件 | `find . -name "*.wav"` |

## 权限管理

| 命令 | 功能 | 示例 |
|------|------|------|
| `chmod` | 修改权限 | `chmod +x script.sh` |
| `chown` | 修改所有者 | `chown user:user file.txt` |

权限数字：4=读(r), 2=写(w), 1=执行(x)
- `chmod 755 script.sh` = rwxr-xr-x
- `chmod 644 file.txt` = rw-r--r--

## 管道与重定向

| 符号 | 功能 | 示例 |
|------|------|------|
| `|` | 管道：将前一命令输出传给下一命令 | `cat log.txt \| grep "error"` |
| `>` | 重定向输出（覆盖） | `python train.py > log.txt` |
| `>>` | 重定向输出（追加） | `echo "done" >> log.txt` |
| `2>` | 重定向错误输出 | `python train.py 2> error.log` |
| `&>` | 重定向所有输出 | `python train.py &> all.log` |

## 文本处理

| 命令 | 功能 | 示例 |
|------|------|------|
| `grep` | 搜索文本 | `grep "error" log.txt` |
| `grep -r` | 递归搜索 | `grep -r "import torch" src/` |
| `grep -i` | 忽略大小写 | `grep -i "Error" log.txt` |
| `sed` | 文本替换 | `sed 's/old/new/g' file.txt` |
| `awk` | 文本处理 | `awk '{print $1, $3}' data.csv` |
| `sort` | 排序 | `sort -k2 -n data.csv` |
| `uniq` | 去重 | `sort data.txt \| uniq` |
| `cut` | 提取列 | `cut -d',' -f1,3 data.csv` |
| `tr` | 字符替换 | `echo "Hello" \| tr 'A-Z' 'a-z'` |

## 进程管理

| 命令 | 功能 | 示例 |
|------|------|------|
| `ps` | 查看进程 | `ps aux` |
| `top` / `htop` | 实时监控 | `htop` |
| `kill` | 终止进程 | `kill 12345` |
| `kill -9` | 强制终止 | `kill -9 12345` |
| `nvidia-smi` | 查看GPU状态 | `nvidia-smi` |
| `bg` | 后台运行 | `ctrl+z` 然后 `bg` |
| `nohup` | 断开SSH后继续运行 | `nohup python train.py &` |

## SSH与远程连接

| 命令 | 功能 | 示例 |
|------|------|------|
| `ssh` | 远程连接 | `ssh user@192.168.1.100` |
| `ssh -p` | 指定端口 | `ssh -p 2222 user@server` |
| `scp` | 远程复制文件 | `scp file.txt user@server:~/data/` |
| `scp -r` | 远程复制目录 | `scp -r data/ user@server:~/backup/` |

## tmux（终端复用）

```bash
# 新建session
tmux new -s train

# 分离session（不断开SSH程序继续运行）
# 按键: Ctrl+b 然后按 d

# 重新连接
tmux attach -s train

# 列出所有session
tmux ls

# 在session中新建窗口
# 按键: Ctrl+b 然后按 c

# 切换窗口
# 按键: Ctrl+b 然后按 0/1/2...
```

## 音频处理常用命令

```bash
# 查看音频信息
soxi file.wav
ffprobe file.wav

# 转换采样率
sox input.wav -r 16000 output.wav

# 转换格式
ffmpeg -i input.mp3 output.wav

# 批量转换
for f in *.mp3; do ffmpeg -i "$f" "${f%.mp3}.wav"; done

# 截取片段
sox input.wav output.wav trim 1.0 2.0    # 从1.0秒开始截取2.0秒

# 拼接音频
sox part1.wav part2.wav combined.wav

# 调整音量
sox input.wav output.wav vol 0.5           # 音量减半

# 混合两段音频
sox -m speech.wav noise.wav mixed.wav

# 生成白噪声
sox -n noise.wav synth 5.0 whitenoise
```

## Python环境管理

```bash
# Conda常用命令
conda create -n myenv python=3.10          # 创建环境
conda activate myenv                        # 激活环境
conda deactivate                            # 退出环境
conda env list                              # 列出所有环境
conda env export > environment.yml          # 导出环境配置
conda env create -f environment.yml         # 从配置创建环境
conda install numpy scipy matplotlib         # 安装包
pip install -r requirements.txt             # 用pip安装

# 查看已安装的包
conda list
pip list
```

## 实用技巧

```bash
# 查看文件大小
du -sh data/                   # 查看data目录总大小
du -sh * | sort -hr            # 当前目录下所有文件/目录大小排序

# 后台运行训练
nohup python train.py --epochs 100 > train.log 2>&1 &

# 查看GPU使用情况
nvidia-smi                      # 查看一次
watch -n 1 nvidia-smi           # 每秒刷新

# 查看端口占用
lsof -i :8888

# 快速统计
find . -name "*.wav" | wc -l    # 统计wav文件数量
```
