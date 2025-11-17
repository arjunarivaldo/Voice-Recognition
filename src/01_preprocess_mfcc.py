# src/01_preprocess_mfcc.py
import os
import librosa 
import numpy as np
from datasets import load_dataset, Audio
from functools import partial
import psutil

# Impor helper dan konfigurasi dari file .py kita
try:
    from utils_asr import clean_text
    import config_asr as config 
except ImportError:
    print("Error: Pastikan file 'utils_asr.py' dan 'config_asr.py' ada di folder 'src/' yang sama.")
    exit()

# --- Membuat pemetaan Tokenizer dari Config ---
# Menggunakan VOCAB dari config_asr.py
char_to_num = {char: i + 1 for i, char in enumerate(config.VOCAB)} # +1 agar 0 bisa jadi <blank>


def preprocess_batch(batch, target_sr, n_mfcc):
    """
    Fungsi utama yang akan di-apply menggunakan .map()
    Mirip dengan _preprocess_batch di Project 2 Anda.
    """
    
    # --- 1. Audio Preprocessing (MFCC) ---
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]
    
    if audio_array is None or len(audio_array) == 0:
        batch["input_features"] = None
        batch["labels"] = None
        return batch

    # Ekstraksi MFCC (menggunakan variabel config)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sampling_rate, n_mfcc=n_mfcc)
    batch["input_features"] = mfcc.T 
    
    # --- 2. Text Preprocessing (Tokenizing) ---
    cleaned_transcription = clean_text(batch["transcription"])
    
    # Tokenisasi (menggunakan char_to_num yang dibuat dari config)
    labels = [char_to_num[char] for char in cleaned_transcription if char in char_to_num]
    
    batch["labels"] = labels
    
    return batch

def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Gunakan variabel dari config
    output_dir = config.OUTPUT_DIR_MFCC
    
    if os.path.exists(output_dir):
        print(f"Folder output '{output_dir}' sudah ada. Preprocessing dilewati.")
        return

    # 1. Load Dataset (menggunakan variabel config)
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")

    # 2. Resampling Audio (menggunakan variabel config)
    print(f"Melakukan resampling audio ke {config.TARGET_SAMPLING_RATE} Hz...")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )

    # 3. Menerapkan Preprocessing (MFCC & Tokenizing)
    print("Memulai preprocessing MFCC dan Tokenisasi Teks...")
    num_cpus = os.cpu_count() or 2
    print(f"Menggunakan {num_cpus} core CPU...")

    # 'partial' untuk "membekukan" argumen
    preprocess_func = partial(
        preprocess_batch,
        target_sr=config.TARGET_SAMPLING_RATE,
        n_mfcc=config.N_MFCC # Gunakan variabel config
    )
    
    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=False, # MFCC lebih mudah di-handle satu per satu
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names # Hapus kolom lama
    )
    
    # Filter data yang mungkin gagal (audionya kosong)
    processed_dataset = processed_dataset.filter(
        lambda x: x["input_features"] is not None and x["labels"] is not None
    )

    # 4. Simpan ke Disk
    print(f"Preprocessing selesai. Menyimpan data ke '{output_dir}'...")
    processed_dataset.save_to_disk(output_dir)
    print(f"Data yang telah diproses berhasil disimpan di '{output_dir}'.")
    print(f"Total data yang diproses: {len(processed_dataset)}")


if __name__ == "__main__":
    main()