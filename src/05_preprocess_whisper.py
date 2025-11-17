# src/05_preprocess_whisper.py
import os
import psutil
from datasets import load_dataset, Audio
from transformers import WhisperProcessor 
from functools import partial

# Impor helper dan konfigurasi
try:
    # TIDAK perlu clean_text()! 
    # Whisper dilatih pada data internet (termasuk tanda baca, kapital, dll).
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di 'src/'.")
    exit()

def preprocess_batch_whisper(batch, processor):
    """
    Fungsi preprocessing untuk Whisper.
    """
    audio_input = batch["audio"]
    
    # --- PENTING ---
    # TIDAK membersihkan teks (no clean_text).
    # Whisper dilatih untuk memprediksi tanda baca, kapital, dll.
    # Menggunakan "transcription" asli.
    transcription_text = batch["transcription"]

    # 1. Audio: Ekstraksi fitur (Log-Mel Spectrogram)
    # Processor.feature_extractor akan mengubah audio array -> "input_features"
    input_features = processor.feature_extractor(
        audio_input["array"], 
        sampling_rate=audio_input["sampling_rate"]
    ).input_features[0] # Ambil [0] karena batched=False
    
    # 2. Teks: Tokenisasi
    # Processor.tokenizer akan mengubah teks -> "labels"
    labels = processor.tokenizer(transcription_text).input_ids

    batch["input_features"] = input_features
    batch["labels"] = labels
    
    return batch

def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    output_dir = config.OUTPUT_DIR_WHISPER_DATA
    
    if os.path.exists(output_dir):
        print(f"Folder output '{output_dir}' sudah ada. Preprocessing dilewati.")
        return

    # 1. Load Processor (Tokenizer + Feature Extractor)
    print(f"Memuat Processor dari '{config.WHISPER_MODEL_NAME}'...")
    processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)

    # 2. Load Dataset
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")

    # 3. Resampling Audio (WAJIB 16kHz untuk Whisper)
    print(f"Melakukan resampling audio ke {config.TARGET_SAMPLING_RATE} Hz...")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )

    # 4. Menerapkan Preprocessing
    print("Memulai preprocessing Whisper (Feature Extraction + Tokenization)...")
    num_cpus = os.cpu_count() or 2
    print(f"Menggunakan {num_cpus} core CPU...")

    preprocess_func = partial(
        preprocess_batch_whisper,
        processor=processor
    )
    
    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=False, 
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names # Hapus kolom lama
    )
    
    # 5. Simpan ke Disk
    print(f"Preprocessing selesai. Menyimpan data ke '{output_dir}'...")
    processed_dataset.save_to_disk(output_dir)
    print(f"Data yang telah diproses berhasil disimpan di '{output_dir}'.")
    print(f"Total data yang diproses: {len(processed_dataset)}")


if __name__ == "__main__":
    main()