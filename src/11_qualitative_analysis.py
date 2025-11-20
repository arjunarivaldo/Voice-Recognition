# src/11_qualitative_analysis.py
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# Impor konfigurasi
try:
    import config_asr as config
except ImportError:
    print("Error: Pastikan 'config_asr.py' ada di folder 'src/'.")
    exit()

# --- 1. Fungsi Prediksi Whisper  ---

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

# --- 2. Fungsi Normalisasi Teks Sederhana ---
def normalize_text(text):
    """Normalisasi sederhana untuk perbandingan yang adil."""
    if not isinstance(text, str):
        return ""
    # Hapus tanda baca, ubah ke lowercase, hapus spasi ganda
    text = text.lower()
    text = text.replace(".", "").replace(",", "").replace("?", "").replace("'", "")
    text = " ".join(text.split()) # Hapus spasi ganda/awal/akhir
    return text

# --- 3. Fungsi Main Analisis ---
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {DEVICE}")
    
    # (Perbaikan Path) Membuat folder output jika belum ada
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # 1. Muat Data Test (RAW)
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    test_dataset = split_dataset["test"]
    print(f"Data test yang akan dievaluasi: {len(test_dataset)} sampel.")

    # 2. Muat Model ASR Terbaik (Whisper)
    print("Memuat model ASR (Whisper)...")
    whisper_model_path = f"{config.OUTPUT_DIR_WHISPER_MODEL}/best_model"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(DEVICE).eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # 3. Loop Evaluasi
    print("\n--- Memulai Analisis Kesalahan Kualitatif ---")
    
    results = [] # List untuk menyimpan dictionary hasil

    for item in tqdm(test_dataset, desc="Menganalisis Prediksi"):
        audio_data = item["audio"]
        reference = item["transcription"] # Jawaban asli (target)
        
        # Jalankan prediksi Whisper
        prediction = predict_whisper(
            audio_data["array"], 
            audio_data["sampling_rate"], 
            whisper_model, 
            whisper_processor,
            DEVICE
        )
        
        # Normalisasi untuk perbandingan
        ref_normalized = normalize_text(reference)
        pred_normalized = normalize_text(prediction)
        
        # Cek jika ada error
        is_error = (ref_normalized != pred_normalized)
        
        results.append({
            "Reference (Asli)": reference,
            "Prediction (Whisper)": prediction,
            "Is_Error": is_error,
            "Reference (Normalized)": ref_normalized,
            "Prediction (Normalized)": pred_normalized
        })

    # 4. Buat DataFrame dan Simpan Kesalahan
    df = pd.DataFrame(results)
    
    # Filter hanya yang error
    df_errors = df[df["Is_Error"] == True]
    
    output_file = config.QUALITATIVE_REPORT_FILE
    df_errors.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 5. Cetak Hasil
    total_samples = len(df)
    total_errors = len(df_errors)
    ser = total_errors / total_samples # Menghitung SER
    
    print("\n" + "="*55)
    print("--- HASIL AKHIR ANALISIS KESALAHAN KUALITATIF ---")
    print(f"Total Sampel Test: {total_samples}")
    print(f"Total Prediksi Salah (Error): {total_errors}")
    print(f"Sentence Error Rate (SER) Manual: {ser:.4f} (atau {ser*100:.1f}%)")
    print(f"\nFile analisis kesalahan disimpan di: '{output_file}'")
    print("="*55)


if __name__ == "__main__":
    main()