#!/usr/bin/env python3
"""
生成不同 SNR 的语音增强测试样本。
使用 DeepFilterNet assets 目录中的音频文件。
"""
import numpy as np
import soundfile as sf
import os

SR = 48000
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
DURATION = 5.0  # 秒

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE5_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(MODULE5_DIR, "DeepFilterNet-main", "assets")
OUTPUT_DIR = os.path.join(MODULE5_DIR, "test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(path, sr=SR, duration=DURATION):
    """加载音频并截取指定时长"""
    audio, orig_sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if orig_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    target_len = int(sr * duration)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.tile(audio, (target_len // len(audio)) + 1)[:target_len]
    return audio

def mix_at_snr(clean, noise, snr_db):
    """按指定 SNR 混合"""
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return clean.copy()
    snr_linear = 10 ** (snr_db / 10)
    noise_scaled = noise * np.sqrt(clean_power / (noise_power * snr_linear + 1e-8))
    mixture = clean + noise_scaled
    mixture = mixture / (np.max(np.abs(mixture)) + 1e-8) * 0.9
    return mixture

def main():
    clean_path = os.path.join(ASSETS_DIR, "clean_freesound_33711.wav")
    noise_path = os.path.join(ASSETS_DIR, "noise_freesound_573577.wav")

    if not os.path.exists(clean_path):
        print("错误：找不到 assets/clean_freesound_33711.wav")
        print("请确认 DeepFilterNet-main 已完整下载")
        return

    clean = load_audio(clean_path)
    noise = load_audio(noise_path)

    sf.write(os.path.join(OUTPUT_DIR, "clean.wav"), clean, SR)
    print("已保存: clean.wav")

    for snr in SNR_LEVELS:
        mixture = mix_at_snr(clean, noise, snr)
        fname = "noisy_snr%d.wav" % snr
        sf.write(os.path.join(OUTPUT_DIR, fname), mixture.astype(np.float32), SR)
        print("已保存: %s" % fname)

    print("\n完成！测试样本保存在: %s" % OUTPUT_DIR)

if __name__ == "__main__":
    main()
