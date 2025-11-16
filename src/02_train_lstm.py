# src/02_train_lstm.py
import os
import torch
import numpy as np
import psutil
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

from datasets import load_from_disk, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor, # Kita pinjam processor-nya hanya untuk decode metric
)

# Impor konfigurasi dan model BARU kita
try:
    import config_asr as config
    from model_asr import LstmCtcConfig, LstmCtcForAsr
except ImportError:
    print("Error: Pastikan 'config_asr.py' dan 'model_asr.py' ada di folder 'src/'.")
    exit()


# --- 1. Data Collator (DI-UPGRADE) ---
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator yang di-upgrade untuk padding DAN menghitung lengths.
    """
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # --- Input (MFCC) & Input Lengths ---
        input_features = [{"input_features": f["input_features"]} for f in features]
        
        # (BARU) Simpan panjang asli (unpadded) dari setiap MFCC
        input_lengths = [len(f["input_features"]) for f in input_features]
        
        max_input_len = max(input_lengths)
        
        batch_inputs = torch.zeros(
            (len(input_features), max_input_len, config.N_MFCC),
            dtype=torch.float
        )
        
        # (BARU) Buat Attention Mask (1 untuk data asli, 0 untuk padding)
        # Mirip Project 2
        attention_mask = torch.zeros(
            (len(input_features), max_input_len), 
            dtype=torch.long
        )

        for i, f in enumerate(input_features):
            mfccs = f["input_features"]
            batch_inputs[i, :len(mfccs), :] = torch.tensor(mfccs, dtype=torch.float)
            attention_mask[i, :len(mfccs)] = 1 # Set 1 untuk data asli
            
        # --- Label (Teks) & Label Lengths ---
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        # (BARU) Simpan panjang asli (unpadded) dari setiap label
        label_lengths = [len(f["input_ids"]) for f in label_features]
        
        max_label_len = max(label_lengths)
        
        batch_labels = torch.full(
            (len(label_features), max_label_len), 
            fill_value=-100, # Ignore index
            dtype=torch.long
        )
        
        for i, f in enumerate(label_features):
            labels = f["input_ids"]
            batch_labels[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
            
        # Kembalikan dictionary lengkap
        return {
            "input_features": batch_inputs,
            "labels": batch_labels,
            "attention_mask": attention_mask, # (BARU) Untuk model
            "input_lengths": torch.tensor(input_lengths, dtype=torch.long), # (BARU) Untuk model
            "label_lengths": torch.tensor(label_lengths, dtype=torch.long)  # (BARU) Untuk model
        }

# --- 2. Fungsi Metrik (BARU) ---
# Kita perlu cara untuk decode prediksi angka -> teks
# Kita buat "tokenizer" palsu dari vocab kita
class SimpleCtcTokenizer:
    def __init__(self, vocab_list, blank_id=0):
        # map angka ke karakter (1 -> ' ', 2 -> 'a', ...)
        self.num_to_char = {i + 1: char for i, char in enumerate(vocab_list)}
        self.blank_id = blank_id
        
    def decode(self, token_ids):
        # Fungsi ini akan menggabungkan & menghapus token blank/duplikat
        # (Sederhana, BUKAN implementasi CTC decode penuh, tapi cukup untuk WER)
        text = ""
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item()
            if token_id == self.blank_id: # 1. Lewati <blank>
                continue
            if i > 0 and token_id == token_ids[i-1].item(): # 2. Lewati duplikat
                continue
            
            # Tambahkan karakter jika ada di vocab
            if token_id in self.num_to_char:
                text += self.num_to_char[token_id]
        return text

# Inisialisasi tokenizer dan metrik WER
tokenizer_for_metrics = SimpleCtcTokenizer(config.VOCAB, config.CTC_BLANK_TOKEN_ID)
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    # Dapatkan logits (prediksi) dan labels (target)
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # --- PERBAIKAN ---
    # pred.label_ids adalah tuple (labels, input_lengths, label_lengths)
    # Kita hanya ambil elemen pertamanya (labels)
    label_ids = pred.label_ids[0] 
    
    # (Opsional tapi lebih aman) Buat salinan agar bisa diubah
    label_ids = label_ids.copy()
    # ------------------
    
    # Ganti -100 (padding) di labels dengan <blank> (0) agar bisa di-decode
    label_ids[label_ids == -100] = config.CTC_BLANK_TOKEN_ID # (Sekarang aman)
    
    # Decode (Angka -> Teks)
    pred_str = [tokenizer_for_metrics.decode(ids) for ids in pred_ids]
    label_str = [tokenizer_for_metrics.decode(ids) for ids in label_ids]
    
    # Hitung WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


# --- 3. Fungsi Main ---
def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # 1. Muat data yang sudah diproses (dari Langkah 8.2)
    print(f"Memuat data MFCC yang diproses dari '{config.OUTPUT_DIR_MFCC}'...")
    if not os.path.exists(config.OUTPUT_DIR_MFCC):
        print(f"Error: Folder data '{config.OUTPUT_DIR_MFCC}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan ulang 'src/01_preprocess_mfcc.py' dengan tokenizer +1.")
        return
        
    processed_dataset = load_from_disk(config.OUTPUT_DIR_MFCC)

    # 2. Split Data (90/10)
    print("Membuat split Train/Validation (90/10)...")
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    split_dataset["validation"] = split_dataset.pop("test")
    
    print(f"Jumlah data train: {len(split_dataset['train'])}")
    print(f"Jumlah data validation: {len(split_dataset['validation'])}")

    # 3. Inisialisasi Data Collator
    data_collator = DataCollatorCTCWithPadding()

    # 4. Inisialisasi Model & Config (BARU)
    print("Memuat Config dan Model (LSTM)...")
    model_config = LstmCtcConfig(
        input_size=config.LSTM_INPUT_SIZE,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
        vocab_size=config.CTC_VOCAB_SIZE, # 28
        ctc_blank_token_id=config.CTC_BLANK_TOKEN_ID # 0
    )
    
    model = LstmCtcForAsr(config=model_config)
    
    # 5. Inisialisasi Training Arguments
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR_LSTM,
        num_train_epochs=50, # Kita perlu epoch lebih banyak (coba 50)
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        
        logging_dir=f"{config.OUTPUT_DIR_LSTM}/logs",
        logging_strategy="steps",
        logging_steps=10, # Log lebih sering
        
        eval_strategy="epoch", 
        save_strategy="epoch", 
        
        load_best_model_at_end=True,
        metric_for_best_model="wer", # (BARU) Gunakan WER
        greater_is_better=False,    # WER lebih kecil lebih baik
        
        fp16=torch.cuda.is_available(), 
        report_to="tensorboard",
        disable_tqdm=True, 
        
        # (BARU) Penting untuk CTC: abaikan 'labels' dari input model
        # karena kita sudah menanganinya di dalam 'forward'
        label_names=["labels", "input_lengths", "label_lengths"]
    )

    # 6. Inisialisasi Trainer (LENGKAP)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics, # (BARU)
    )

    # 7. Mulai Training
    print("--- MEMULAI TRAINING LSTM ---")
    trainer.train()
    print("--- TRAINING LSTM SELESAI ---")

    # 8. Simpan model terbaik
    print(f"Menyimpan model terbaik ke {config.OUTPUT_DIR_LSTM}/best_model ...")
    trainer.save_model(f"{config.OUTPUT_DIR_LSTM}/best_model")
    print("Model berhasil disimpan.")

if __name__ == "__main__":
    main()