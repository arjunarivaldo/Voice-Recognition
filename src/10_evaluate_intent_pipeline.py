# src/10_evaluate_intent_pipeline.py
import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

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

# --- 1. Fungsi Prediksi (Diambil dari skrip evaluasi sebelumnya) ---

@torch.no_grad()
def predict_whisper(audio_array, sampling_rate, model, processor, device):
    """
    Menjalankan prediksi ASR (Audio -> Teks) menggunakan Whisper.
    """
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
    """
    Menjalankan prediksi Intent Classification (Teks -> Intensi).
    """
    # Tokenisasi teks (hasil dari Whisper)
    inputs = tokenizer(
        text,
        max_length=config.INTENT_MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    # Prediksi
    logits = model(**inputs).logits
    
    # Ambil kelas dengan probabilitas tertinggi
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
    
    # Resample ke 16kHz (standar)
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )
    
    # Ambil 10% data yang SAMA PERSIS dengan yang digunakan untuk validasi
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    test_dataset = split_dataset["test"]
    print(f"Data test yang akan dievaluasi: {len(test_dataset)} sampel.")

    # 3. Muat Model-Model Terbaik
    
    # A. Model ASR (Whisper)
    print("Memuat model ASR (Whisper)...")
    whisper_model_path = f"{config.OUTPUT_DIR_WHISPER_MODEL}/best_model"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(DEVICE).eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # B. Model Intent Classification (BERT)
    print("Memuat model Intent Classification (BERT)...")
    intent_model_path = f"{config.OUTPUT_DIR_INTENT_MODEL}/best_model"
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path).to(DEVICE).eval()
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)

    # 4. Loop Evaluasi Bertingkat (Cascaded)
    print("\n--- Memulai Evaluasi Pipeline Bertingkat (SLU) ---")
    
    true_intents = []
    predicted_intents = []

    for item in tqdm(test_dataset, desc="Mengevaluasi Pipeline SLU"):
        audio_data = item["audio"]
        
        # Ambil jawaban intensi yang benar (target)
        true_intent_id = item["intent_class"]
        true_intents.append(true_intent_id)
        
        # --- LANGKAH A: ASR (Audio -> Teks) ---
        predicted_text = predict_whisper(
            audio_data["array"], 
            audio_data["sampling_rate"], 
            whisper_model, 
            whisper_processor,
            DEVICE
        )
        
        # --- LANGKAH B: Intent (Teks -> Intensi) ---
        predicted_intent_id = predict_intent(
            predicted_text, 
            intent_model, 
            intent_tokenizer,
            DEVICE
        )
        predicted_intents.append(predicted_intent_id)
        
        # (Opsional: cetak untuk melihat perbandingan)
        print(f"Audio -> Teks: '{predicted_text}'")
        print(f"Teks -> Intensi: {predicted_intent_id} (Target: {true_intent_id}) \n")

    # 5. Hitung Skor Akhir
    print("Menghitung skor Akurasi akhir...")
    
    accuracy = accuracy_metric.compute(predictions=predicted_intents, references=true_intents)
    
    print("\n" + "="*55)
    print("--- HASIL AKHIR AKURASI INTENT CLASSIFICATION ---")
    print(f"(Dataset: {config.LANG_SUBSET}, Test Samples: {len(test_dataset)})")
    print("="*55)
    
    print(f"  Skenario 1 (Ideal):      Teks Asli -> BERT \tAkurasi = 0.9474 (94.7%)")
    print(f"  Skenario 2 (Real-World): Audio -> Whisper -> BERT \tAkurasi = {accuracy['accuracy']:.4f} (atau {accuracy['accuracy']*100:.1f}%)")
    
    print("="*55)
    print("\nEvaluasi selesai.")


if __name__ == "__main__":
    main()