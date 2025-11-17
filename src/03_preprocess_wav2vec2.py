# src/03_preprocess_wav2vec2.py
import os
import psutil
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
from functools import partial

# Impor helper dan konfigurasi
try:
    from utils_asr import clean_text
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'utils_asr.py' dan 'config_asr.py' ada di 'src/'.")
    exit()

def preprocess_batch_wav2vec2(batch, processor):
    """
    Fungsi preprocessing untuk Wav2Vec2 (DIPERBAIKI).
    """
    audio_input = batch["audio"]
    cleaned_text = clean_text(batch["transcription"])

    # Panggil processor dengan [array] dan [text] untuk memaksanya 
    # memproses sebagai "batch-of-1"
    processed_batch = processor(
        [audio_input["array"]], 
        sampling_rate=audio_input["sampling_rate"], 
        text=[cleaned_text]
    )
    
    batch["input_values"] = processed_batch["input_values"][0] 
    batch["labels"] = processed_batch["labels"][0]
    
    return batch


def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    output_dir = config.OUTPUT_DIR_WAV2VEC2_DATA
    
    if os.path.exists(output_dir):
        print(f"Folder output '{output_dir}' sudah ada. Preprocessing dilewati.")
        return

    # 1. Load Processor (Tokenizer + Feature Extractor)
    print(f"Memuat Processor dari '{config.WAV2VEC2_MODEL_NAME}'...")
    # Penting: pastikan token <blank> (untuk CTC) adalah token 'pad'
    processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC2_MODEL_NAME)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 2. Load Dataset
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")

    # 3. Resampling Audio
    print(f"Melakukan resampling audio ke {config.TARGET_SAMPLING_RATE} Hz...")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )

    # 4. Menerapkan Preprocessing
    print("Memulai preprocessing Wav2Vec2...")
    num_cpus = os.cpu_count() or 2
    print(f"Menggunakan {num_cpus} core CPU...")

    preprocess_func = partial(
        preprocess_batch_wav2vec2,
        processor=processor
    )
    
    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=False, 
        num_proc=num_cpus,
        remove_columns=raw_dataset.column_names # Hapus kolom lama
    )
    
    # Filter data yang mungkin teksnya kosong setelah di-clean
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["labels"]) > 0
    )

    # 5. Simpan ke Disk
    print(f"Preprocessing selesai. Menyimpan data ke '{output_dir}'...")
    processed_dataset.save_to_disk(output_dir)
    print(f"Data yang telah diproses berhasil disimpan di '{output_dir}'.")
    print(f"Total data yang diproses: {len(processed_dataset)}")


if __name__ == "__main__":
    main()