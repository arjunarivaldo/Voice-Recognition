# src/10_evaluate_intent_pipeline.py
import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

# --- Impor untuk plotting ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# -----------------------------------

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

# --- 1. Fungsi Prediksi ---

@torch.no_grad()
def predict_whisper(audio_array, sampling_rate, model, processor, device):
    input_features = processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(device)
    predicted_ids = model.generate(input_features, max_length=128)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

@torch.no_grad()
def predict_intent(text, model, tokenizer, device):
    inputs = tokenizer(
        text,
        max_length=config.INTENT_MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

# --- 2. Fungsi Main Evaluasi Pipeline ---
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {DEVICE}")

    # 1. Muat Metrik Akurasi
    accuracy_metric = evaluate.load("accuracy")

    # 2. Muat Data Test (RAW)
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    test_dataset = split_dataset["test"]
    print(f"Data test yang akan dievaluasi: {len(test_dataset)} sampel.")

    # 3. Muat Model-Model Terbaik
    print("Memuat model ASR (Whisper)...")
    whisper_model_path = f"{config.OUTPUT_DIR_WHISPER_MODEL}/best_model"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(DEVICE).eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)

    print("Memuat model Intent Classification (BERT)...")
    intent_model_path = f"{config.OUTPUT_DIR_INTENT_MODEL}/best_model"
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path).to(DEVICE).eval()
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)

    # 4. Loop Evaluasi
    true_intents = []
    predicted_intents_ideal = []    
    predicted_intents_realworld = [] 

    print("\n--- Memulai Evaluasi Pipeline Bertingkat (SLU) ---")
    for item in tqdm(test_dataset, desc="Mengevaluasi Pipeline SLU"):
        audio_data = item["audio"]
        true_text = item["transcription"]
        true_intent_id = item["intent_class"]
        
        true_intents.append(true_intent_id)
        
        # --- SKENARIO 1: IDEAL (Teks Asli -> BERT) ---
        pred_intent_ideal = predict_intent(
            true_text, 
            intent_model, 
            intent_tokenizer,
            DEVICE
        )
        predicted_intents_ideal.append(pred_intent_ideal)
        
        # --- SKENARIO 2: REAL-WORLD (Audio -> Whisper -> BERT) ---
        predicted_text = predict_whisper(
            audio_data["array"], 
            audio_data["sampling_rate"], 
            whisper_model, 
            whisper_processor,
            DEVICE
        )
        pred_intent_realworld = predict_intent(
            predicted_text, 
            intent_model, 
            intent_tokenizer,
            DEVICE
        )
        predicted_intents_realworld.append(pred_intent_realworld)

    # 5. Hitung Skor Akhir
    print("Menghitung skor Akurasi akhir...")
    
    accuracy_ideal = accuracy_metric.compute(predictions=predicted_intents_ideal, references=true_intents)
    accuracy_realworld = accuracy_metric.compute(predictions=predicted_intents_realworld, references=true_intents)
    
    # 6. Cetak Hasil Teks
    print("\n" + "="*55)
    print("--- HASIL AKHIR AKURASI INTENT CLASSIFICATION ---")
    print(f"(Dataset: {config.LANG_SUBSET}, Test Samples: {len(test_dataset)})")
    print("="*55)
    
    print(f"  Skenario 1 (Ideal):      Teks Asli -> BERT \tAkurasi = {accuracy_ideal['accuracy']:.4f} (atau {accuracy_ideal['accuracy']*100:.1f}%)")
    print(f"  Skenario 2 (Real-World): Audio -> Whisper -> BERT \tAkurasi = {accuracy_realworld['accuracy']:.4f} (atau {accuracy_realworld['accuracy']*100:.1f}%)")
    
    print("="*55)

    # 7. Visualisasi
    print("\nMembuat visualisasi perbandingan Akurasi...")
    
    # Siapkan data
    data = {
        'Skenario': [
            'Skenario 1 (Ideal) Teks Asli -> BERT', 
            'Skenario 2 (Real-World) Audio -> Whisper -> BERT'
        ],
        'Akurasi': [accuracy_ideal['accuracy'], accuracy_realworld['accuracy']]
    }
    df_results = pd.DataFrame(data)
    
    # Buat Bar Plot
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        x='Skenario', 
        y='Akurasi', 
        data=df_results,
        palette='rocket' # Menggunakan palet warna
    )
    
    # Tambahkan label/judul
    plt.title('Perbandingan Akurasi Klasifikasi Intensi (Pipeline SLU)', fontsize=16, pad=20)
    plt.xlabel('Skenario Pipeline', fontsize=12)
    plt.ylabel('Akurasi (Lebih tinggi lebih baik)', fontsize=12)
    plt.ylim(0, 1.0) # Akurasi dari 0 sampai 1
    
    # Tambahkan nilai (angka) di atas setiap bar
    for p in barplot.patches:
            barplot.annotate(f'{p.get_height():.4f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center',  
                        xytext=(0, 9),        
                        textcoords='offset points',
                        fontsize=11,)
    plt.tight_layout()
    
    # Simpan file plot
    plot_filename = 'intent_accuracy_comparison.png'
    plt.savefig(plot_filename)
    print(f"Visualisasi berhasil disimpan ke '{plot_filename}'")
    
    print("\nEvaluasi selesai.")


if __name__ == "__main__":
    main()