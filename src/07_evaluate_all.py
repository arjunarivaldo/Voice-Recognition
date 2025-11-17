# src/07_evaluate_all.py
import os
import torch
import librosa
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from tqdm.auto import tqdm

# Impor untuk plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# Impor config, model LSTM, dan utils
try:
    import config_asr as config
    from model_asr import LstmCtcConfig, LstmCtcForAsr
    from utils_asr import clean_text
except ImportError:
    print("Error: Pastikan 'config_asr.py', 'model_asr.py', dan 'utils_asr.py' ada di 'src/'.")
    exit()

# --- 1. Helper Tokenizer untuk LSTM ---
class SimpleCtcTokenizer:
    def __init__(self, vocab_list, blank_id=0):
        self.num_to_char = {i + 1: char for i, char in enumerate(vocab_list)}
        self.blank_id = blank_id
        
    def decode(self, token_ids):
        text = ""
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item()
            if token_id == self.blank_id or (i > 0 and token_id == token_ids[i-1].item()):
                continue
            if token_id in self.num_to_char:
                text += self.num_to_char[token_id]
        return text

# --- 2. Fungsi Prediksi (Tidak Berubah) ---

@torch.no_grad() 
def predict_lstm(audio_array, sampling_rate, model, tokenizer):
    mfcc = librosa.feature.mfcc(
        y=audio_array, 
        sr=sampling_rate, 
        n_mfcc=config.N_MFCC
    ).T
    input_features = torch.tensor(mfcc, dtype=torch.float).to(model.device)
    input_features = input_features.unsqueeze(0) 
    logits = model(input_features=input_features).logits
    pred_ids = torch.argmax(logits, dim=-1)[0] 
    return tokenizer.decode(pred_ids)

@torch.no_grad()
def predict_wav2vec2(audio_array, sampling_rate, model, processor):
    input_values = processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_values.to(model.device)
    logits = model(input_values=input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    return processor.decode(pred_ids)

@torch.no_grad()
def predict_whisper(audio_array, sampling_rate, model, processor):
    input_features = processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(model.device)
    predicted_ids = model.generate(input_features, max_length=128)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# --- Fungsi Manual untuk SER ---
def calculate_ser(predictions, references):
    """
    Menghitung Sentence Error Rate (SER) secara manual.
    SER = (jumlah kalimat yang tidak sama persis) / (total kalimat)
    """
    if len(predictions) == 0:
        return 0.0
    errors = 0
    for pred, ref in zip(predictions, references):
        # Gunakan .strip() untuk menghapus spasi di awal/akhir
        if pred.strip() != ref.strip():
            errors += 1
    return errors / len(predictions)
# ------------------------------------

# --- 3. Fungsi Main Evaluasi ---
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {DEVICE}")

    # 1. Muat Metrik 
    print("Memuat metrik WER, CER, dan BERTScore...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    bertscore_metric = evaluate.load("bertscore")

    # 2. Muat Data Test (RAW)
    print(f"Memuat dataset '{config.DATASET_NAME}' subset '{config.LANG_SUBSET}'...")
    raw_dataset = load_dataset(config.DATASET_NAME, config.LANG_SUBSET, split="train")
    raw_dataset = raw_dataset.cast_column(
        "audio", Audio(sampling_rate=config.TARGET_SAMPLING_RATE)
    )
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
    test_dataset = split_dataset["test"]
    print(f"Data test yang akan dievaluasi: {len(test_dataset)} sampel.")

    # 3. Muat Model & Processor
    print("Memuat model LSTM...")
    lstm_model_path = f"{config.OUTPUT_DIR_LSTM}/best_model"
    lstm_model = LstmCtcForAsr.from_pretrained(lstm_model_path).to(DEVICE).eval()
    lstm_tokenizer = SimpleCtcTokenizer(config.VOCAB, config.CTC_BLANK_TOKEN_ID)

    print("Memuat model Wav2Vec2...")
    w2v2_model_path = f"{config.OUTPUT_DIR_WAV2VEC2_MODEL}/best_model"
    w2v2_model = Wav2Vec2ForCTC.from_pretrained(w2v2_model_path).to(DEVICE).eval()
    w2v2_processor = Wav2Vec2Processor.from_pretrained(w2v2_model_path)

    print("Memuat model Whisper...")
    whisper_model_path = f"{config.OUTPUT_DIR_WHISPER_MODEL}/best_model"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to(DEVICE).eval()
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_path)

    # 4. Loop Evaluasi
    print("\n--- Memulai Evaluasi Komparatif ---")
    references_cleaned = [] 
    references_original = [] 
    preds_lstm = []
    preds_w2v2 = []
    preds_whisper = []

    for item in tqdm(test_dataset, desc="Mengevaluasi"):
        audio_data = item["audio"]
        ref_original = item["transcription"]
        ref_cleaned = clean_text(ref_original)
        
        references_original.append(ref_original)
        references_cleaned.append(ref_cleaned)
        
        preds_lstm.append(
            predict_lstm(audio_data["array"], audio_data["sampling_rate"], lstm_model, lstm_tokenizer)
        )
        preds_w2v2.append(
            predict_wav2vec2(audio_data["array"], audio_data["sampling_rate"], w2v2_model, w2v2_processor)
        )
        preds_whisper.append(
            predict_whisper(audio_data["array"], audio_data["sampling_rate"], whisper_model, whisper_processor)
        )

    # 5. Hitung Skor Akhir (DI-UPGRADE)
    print("Menghitung skor WER, CER, SER...")
    
    # Hitung WER & CER (dari library)
    wer_lstm = wer_metric.compute(predictions=preds_lstm, references=references_cleaned)
    cer_lstm = cer_metric.compute(predictions=preds_lstm, references=references_cleaned)
    wer_w2v2 = wer_metric.compute(predictions=preds_w2v2, references=references_cleaned)
    cer_w2v2 = cer_metric.compute(predictions=preds_w2v2, references=references_cleaned)
    wer_whisper = wer_metric.compute(predictions=preds_whisper, references=references_original)
    cer_whisper = cer_metric.compute(predictions=preds_whisper, references=references_original)

    # Hitung SER (Manual)
    ser_lstm = calculate_ser(preds_lstm, references_cleaned)
    ser_w2v2 = calculate_ser(preds_w2v2, references_cleaned)
    ser_whisper = calculate_ser(preds_whisper, references_original)
    
    print("Menghitung BERTScore (mungkin perlu beberapa saat)...")
    bertscore_lstm = bertscore_metric.compute(predictions=preds_lstm, references=references_cleaned, lang="en")
    bertscore_w2v2 = bertscore_metric.compute(predictions=preds_w2v2, references=references_cleaned, lang="en")
    bertscore_whisper = bertscore_metric.compute(predictions=preds_whisper, references=references_original, lang="en")

    bs_lstm_f1 = np.mean(bertscore_lstm['f1'])
    bs_w2v2_f1 = np.mean(bertscore_w2v2['f1'])
    bs_whisper_f1 = np.mean(bertscore_whisper['f1'])

    # 6. Cetak Hasil Teks
    print("\n" + "="*55)
    print("--- HASIL AKHIR EVALUASI KOMPARATIF ---")
    print(f"(Dataset: {config.LANG_SUBSET}, Test Samples: {len(test_dataset)})")
    print("="*55)
    
    models = ["LSTM from 0", "Wav2Vec2 Fine-Tune", "Whisper Fine-Tune"]
    results_wer = [wer_lstm, wer_w2v2, wer_whisper]
    results_cer = [cer_lstm, cer_w2v2, cer_whisper]
    results_ser = [ser_lstm, ser_w2v2, ser_whisper]
    results_bs = [bs_lstm_f1, bs_w2v2_f1, bs_whisper_f1]
    
    for i, model_name in enumerate(models):
        print(f"\n--- {model_name} ---")
        print(f"  WER (↓): \t{results_wer[i]:.4f} (atau {results_wer[i]*100:.1f}%)")
        print(f"  CER (↓): \t{results_cer[i]:.4f} (atau {results_cer[i]*100:.1f}%)")
        print(f"  SER (↓): \t{results_ser[i]:.4f} (atau {results_ser[i]*100:.1f}%)")
        print(f"  BERT-F1 (↑):\t{results_bs[i]:.4f} (atau {results_bs[i]*100:.1f}%)")
        
    print("="*55)

    # 7. Visualisasi 
    print("\nMembuat visualisasi perbandingan metrik...")
    
    data = {
        'Model': models * 4, 
        'Metric': ['WER'] * 3 + ['CER'] * 3 + ['SER'] * 3 + ['BERTScore_F1'] * 3,
        'Score': results_wer + results_cer + results_ser + results_bs
    }
    df_results = pd.DataFrame(data)
    
    df_errors = df_results[df_results['Metric'].isin(['WER', 'CER', 'SER'])]
    df_bert = df_results[df_results['Metric'] == 'BERTScore_F1']

    # --- Plot 1: Error Rates (WER, CER, SER) ---
    plt.figure(figsize=(12, 7))
    barplot_error = sns.barplot(
        x='Model', 
        y='Score', 
        hue='Metric', 
        data=df_errors,
        palette='muted'
    )
    plt.title('Perbandingan Error Rates (WER, CER, SER)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Error Rate (Lebih rendah lebih baik)', fontsize=12)
    plt.ylim(0, max(df_errors['Score'].max() * 1.1, 1.1)) 
    
    for p in barplot_error.patches:
        barplot_error.annotate(f'{p.get_height():.3f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=9)
    
    plot_filename_1 = 'error_rates_comparison.png'
    plt.savefig(plot_filename_1)
    print(f"Plot Error Rates disimpan ke '{plot_filename_1}'")
    plt.clf()

    # --- Plot 2: BERTScore F1 ---
    plt.figure(figsize=(10, 6))
    barplot_bert = sns.barplot(
        x='Model', 
        y='Score', 
        data=df_bert,
        palette='viridis'
    )
    plt.title('Perbandingan Skor Semantik (BERTScore F1)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('BERTScore F1 (Lebih tinggi lebih baik)', fontsize=12)
    plt.ylim(0, max(df_bert['Score'].max() * 1.1, 0.8)) 
    
    for p in barplot_bert.patches:
        barplot_bert.annotate(f'{p.get_height():.4f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points',
                            fontsize=11)
    
    plot_filename_2 = 'bertscore_comparison.png'
    plt.savefig(plot_filename_2)
    print(f"Plot BERTScore disimpan ke '{plot_filename_2}'")

    print("\nEvaluasi selesai.")

if __name__ == "__main__":
    main()