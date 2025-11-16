# src/06_train_whisper.py
import os
import torch
import numpy as np
import psutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from datasets import load_from_disk, DatasetDict
import evaluate 
from transformers import (
    Trainer,
    TrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration # (BARU) Model Seq2Seq
)

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

# --- 1. Data Collator (BARU untuk Seq2Seq Audio) ---
# Mirip dengan DataCollatorForSeq2Seq, tapi untuk audio "input_features"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator yang menangani padding untuk input audio (features) 
    dan output teks (labels) secara terpisah.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # --- 1. Padding Input (Audio Features) ---
        # "input_features" adalah hasil Log-Mel Spectrogram
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # Gunakan Feature Extractor untuk padding audio
        # Ini akan mem-padding dengan 0.0
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )
        
        # --- 2. Padding Label (Teks) ---
        # "labels" adalah token ID dari tokenizer
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Gunakan Tokenizer untuk padding label
        # Ini akan mem-padding dengan -100 (ignore_index)
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        
        # Ganti pad_token_id (dari tokenizer) dengan -100
        # agar diabaikan oleh loss function
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # --- 3. Gabungkan ---
        batch["labels"] = labels
        return batch

# --- 2. Fungsi Main ---
def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # 1. Muat data yang sudah diproses (dari Langkah 11)
    print(f"Memuat data Whisper yang diproses dari '{config.OUTPUT_DIR_WHISPER_DATA}'...")
    if not os.path.exists(config.OUTPUT_DIR_WHISPER_DATA):
        print(f"Error: Folder data '{config.OUTPUT_DIR_WHISPER_DATA}' tidak ditemukan.")
        print("Jalankan 'src/05_preprocess_whisper.py' terlebih dahulu.")
        return
        
    processed_dataset = load_from_disk(config.OUTPUT_DIR_WHISPER_DATA)

    # 2. Split Data (90/10)
    print("Membuat split Train/Validation (90/10)...")
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    split_dataset["validation"] = split_dataset.pop("test")
    
    print(f"Jumlah data train: {len(split_dataset['train'])}")
    print(f"Jumlah data validation: {len(split_dataset['validation'])}")

    # 3. Inisialisasi Processor dan Model
    print(f"Memuat Processor dan Model Pre-trained '{config.WHISPER_MODEL_NAME}'...")
    processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(config.WHISPER_MODEL_NAME)

    # (Opsional) Whisper tidak perlu 'freeze', tapi kita perlu 
    # mengatur beberapa token ID di config model agar tahu cara .generate()
    model.config.suppress_tokens = []

    # 4. Inisialisasi Data Collator (BARU)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. Inisialisasi Training Arguments
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR_WHISPER_MODEL,
        num_train_epochs=10, # Coba 10 epoch
        per_device_train_batch_size=16, # Model 'tiny' cukup ringan
        per_device_eval_batch_size=16,
        
        logging_dir=f"{config.OUTPUT_DIR_WHISPER_MODEL}/logs",
        logging_strategy="steps",
        logging_steps=10,
        
        eval_strategy="epoch", 
        save_strategy="epoch", 
        
        load_best_model_at_end=True,
        # (PENTING) Kita monitor 'loss', bukan 'wer', seperti di Project 2
        metric_for_best_model="eval_loss", 
        greater_is_better=False, # Loss lebih kecil lebih baik
        
        fp16=torch.cuda.is_available(), 
        report_to="tensorboard",
        disable_tqdm=True, 
    )

    # 6. Inisialisasi Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor, # (PENTING untuk Seq2Seq)
        # Kita tidak pakai compute_metrics, akan dievaluasi di skrip terpisah
    )

    # 7. Mulai Training
    print("--- MEMULAI TRAINING (Fine-Tuning) WHISPER ---")
    trainer.train()
    print("--- TRAINING WHISPER SELESAI ---")

    # 8. Simpan model terbaik
    best_model_path = f"{config.OUTPUT_DIR_WHISPER_MODEL}/best_model"
    print(f"Menyimpan model terbaik ke {best_model_path} ...")
    trainer.save_model(best_model_path)
    # Simpan processor-nya juga
    processor.save_pretrained(best_model_path)
    print("Model dan Processor berhasil disimpan.")

if __name__ == "__main__":
    main()