# src/04_train_wav2vec2.py
import os
import torch
import numpy as np
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

from datasets import load_from_disk, DatasetDict
import evaluate 
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

# --- 1. Inisialisasi Processor & Metrik ---
# Perlu processor untuk decoding metrik
processor = Wav2Vec2Processor.from_pretrained(config.WAV2VEC2_MODEL_NAME)
processor.tokenizer.pad_token = processor.tokenizer.eos_token # Atur token pad
wer_metric = evaluate.load("wer")

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator yang melakukan padding input dan label untuk training CTC.
    (Versi lengkap untuk mengatasi ValueError)
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # --- 1. Padding Input (Audio) ---
        # Pisahkan input_values (audio)
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        # Gunakan Feature Extractor untuk padding audio
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # --- 2. Padding Label (Teks) ---
        # Pisahkan label_features (teks)
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Gunakan Tokenizer untuk padding label
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Ganti pad_token_id (cth: 0) dengan -100 agar diabaikan oleh loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # --- 3. Gabungkan ---
        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    # Dapatkan logits (prediksi) dan labels (target)
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    label_ids = pred.label_ids
    
    # Ganti -100 (padding) di labels dengan token 'pad' dari processor
    # agar bisa di-decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode (Angka -> Teks) menggunakan tokenizer Wav2Vec2
    pred_str = processor.batch_decode(pred_ids)
    # Skip token spesial di label_str
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    
    # Hitung WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}


# --- 2. Fungsi Main ---
def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # 1. Muat data yang sudah diproses (dari Langkah 9)
    print(f"Memuat data Wav2Vec2 yang diproses dari '{config.OUTPUT_DIR_WAV2VEC2_DATA}'...")
    if not os.path.exists(config.OUTPUT_DIR_WAV2VEC2_DATA):
        print(f"Error: Folder data '{config.OUTPUT_DIR_WAV2VEC2_DATA}' tidak ditemukan.")
        print("Jalankan 'src/03_preprocess_wav2vec2.py' terlebih dahulu.")
        return
        
    processed_dataset = load_from_disk(config.OUTPUT_DIR_WAV2VEC2_DATA)

    # 2. Split Data (90/10)
    print("Membuat split Train/Validation (90/10)...")
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    split_dataset["validation"] = split_dataset.pop("test")
    
    print(f"Jumlah data train: {len(split_dataset['train'])}")
    print(f"Jumlah data validation: {len(split_dataset['validation'])}")

    # 3. Inisialisasi Data Collator (Bawaan Transformers)
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        padding=True
    )

    # 4. Inisialisasi Model 
    print(f"Memuat Model Pre-trained '{config.WAV2VEC2_MODEL_NAME}'...")
    model = Wav2Vec2ForCTC.from_pretrained(
        config.WAV2VEC2_MODEL_NAME,
        ctc_loss_reduction="mean", # Tipe loss
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer) # Ukuran vocab dari processor
    )
    
    # --- (PENTING) Freezing Feature Extractor ---
    print("Membekukan (freezing) feature extractor model...")
    model.freeze_feature_extractor()

    # 5. Inisialisasi Training Arguments
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR_WAV2VEC2_MODEL,
        num_train_epochs=10, 
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        
        logging_dir=f"{config.OUTPUT_DIR_WAV2VEC2_MODEL}/logs",
        logging_strategy="steps",
        logging_steps=10,
        
        eval_strategy="epoch", 
        save_strategy="epoch", 
        
        load_best_model_at_end=True,
        metric_for_best_model="wer", # Monitor WER
        greater_is_better=False,     # WER lebih kecil lebih baik
        
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
        compute_metrics=compute_metrics,
    )

    # 7. Mulai Training
    print("--- MEMULAI TRAINING (Fine-Tuning) WAV2VEC2 ---")
    trainer.train()
    print("--- TRAINING WAV2VEC2 SELESAI ---")

    # 8. Simpan model terbaik
    best_model_path = f"{config.OUTPUT_DIR_WAV2VEC2_MODEL}/best_model"
    print(f"Menyimpan model terbaik ke {best_model_path} ...")
    trainer.save_model(best_model_path)
    # Simpan processor-nya juga agar bisa digunakan lagi nanti
    processor.save_pretrained(best_model_path)
    print("Model dan Processor berhasil disimpan.")

if __name__ == "__main__":
    main()