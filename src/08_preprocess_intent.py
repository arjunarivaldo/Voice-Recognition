# src/08_preprocess_intent.py
import os
import psutil
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

def preprocess_batch_intent(batch, tokenizer, max_len):
    """
    Fungsi preprocessing untuk Intent Classification.
    Hanya mentokenisasi 'transcription' dan meneruskan 'intent_class'.
    """
    
    # 1. Tokenisasi Teks (Input)
    # Tidak perlu clean_text, biarkan BERT yang menangani
    tokenized_batch = tokenizer(
        batch["transcription"],
        max_length=max_len,
        truncation=True,
        padding=False # Trainer akan menangani padding dinamis
    )
    
    # 2. Siapkan Label (Target)
    # Ganti nama 'intent_class' menjadi 'labels' agar dikenali Trainer
    tokenized_batch["labels"] = batch["intent_class"]
    
    return tokenized_batch

def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")

    output_dir = config.OUTPUT_DIR_INTENT_DATA
    
    if os.path.exists(output_dir):
        print(f"Folder output '{output_dir}' sudah ada. Preprocessing dilewati.")
        return

    # 1. Load Tokenizer (dari config)
    print(f"Memuat Tokenizer dari '{config.INTENT_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(config.INTENT_MODEL_NAME)

    # 2. Load Dataset
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")

    # 3. Menerapkan Preprocessing
    print("Memulai preprocessing Intent Classification...")
    num_cpus = os.cpu_count() or 2
    print(f"Menggunakan {num_cpus} core CPU...")

    preprocess_func = partial(
        preprocess_batch_intent,
        tokenizer=tokenizer,
        max_len=config.INTENT_MAX_LENGTH
    )
    
    processed_dataset = raw_dataset.map(
        preprocess_func,
        batched=True, 
        num_proc=num_cpus,
        remove_columns=[
            'path', 'audio', 'transcription', 
            'english_transcription', 'intent_class', 'lang_id'
        ]
    )
    
    # 4. Simpan ke Disk
    print(f"Preprocessing selesai. Menyimpan data ke '{output_dir}'...")
    processed_dataset.save_to_disk(output_dir)
    print(f"Data yang telah diproses berhasil disimpan di '{output_dir}'.")
    print(f"Total data yang diproses: {len(processed_dataset)}")


if __name__ == "__main__":
    main()