# src/09_train_intent.py
import os
import torch
import numpy as np
import psutil
import evaluate 

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding # Collator standar untuk padding teks
)

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

# --- 1. Inisialisasi Metrik ---
# Memonitor 'accuracy', 'f1', 'precision', dan 'recall'
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    """
    Menghitung metrik klasifikasi (accuracy, f1, precision, recall).
    Mirip dengan compute_metrics di Project 2.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 'weighted' diperlukan untuk multi-class, 'micro' atau 'macro' juga bisa
    avg_type = "weighted" 
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=avg_type)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=avg_type)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=avg_type)
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }

# --- 2. Fungsi Main ---
def main():
    print(f"Memori RAM tersedia: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # 1. Muat data yang sudah diproses
    print(f"Memuat data Intent yang diproses dari '{config.OUTPUT_DIR_INTENT_DATA}'...")
    if not os.path.exists(config.OUTPUT_DIR_INTENT_DATA):
        print(f"Error: Folder data '{config.OUTPUT_DIR_INTENT_DATA}' tidak ditemukan.")
        print("Jalankan 'src/08_preprocess_intent.py' terlebih dahulu.")
        return
        
    processed_dataset = load_from_disk(config.OUTPUT_DIR_INTENT_DATA)

    # 2. Split Data (90/10)
    print("Membuat split Train/Validation (90/10)...")
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"] 
    
    print(f"Jumlah data train: {len(train_dataset)}")
    print(f"Jumlah data validation: {len(eval_dataset)}")

    # 3. Inisialisasi Tokenizer, Config, dan Model
    print(f"Memuat Tokenizer dan Model dari '{config.INTENT_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(config.INTENT_MODEL_NAME)
    
    model_config = AutoConfig.from_pretrained(
        config.INTENT_MODEL_NAME,
        num_labels=14 
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.INTENT_MODEL_NAME, 
        config=model_config
    )

    # 4. Inisialisasi Data Collator Standar
    # Ini akan mem-padding 'input_ids', 'attention_mask' secara dinamis
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Inisialisasi Training Arguments
    print("Mempersiapkan Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR_INTENT_MODEL,
        num_train_epochs=5, # Klasifikasi teks biasanya cepat konvergen
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        
        logging_dir=f"{config.OUTPUT_DIR_INTENT_MODEL}/logs",
        logging_strategy="steps",
        logging_steps=10,
        
        eval_strategy="epoch", 
        save_strategy="epoch", 
        
        load_best_model_at_end=True,
        metric_for_best_model="accuracy", # Mengunakan 'accuracy'
        greater_is_better=True,
        
        fp16=torch.cuda.is_available(), 
        report_to="tensorboard",
        disable_tqdm=True,
    )

    # 6. Inisialisasi Trainer (STANDAR)
    # Tidak perlu class weights karena datanya balanced
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Mulai Training
    print("--- MEMULAI TRAINING KLASIFIKASI INTENSI ---")
    trainer.train()
    print("--- TRAINING INTENSI SELESAI ---")

    # 8. Simpan model terbaik
    best_model_path = f"{config.OUTPUT_DIR_INTENT_MODEL}/best_model"
    print(f"Menyimpan model terbaik ke {best_model_path} ...")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print("Model dan Tokenizer berhasil disimpan.")

if __name__ == "__main__":
    main()