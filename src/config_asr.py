# src/config_asr.py
"""
File konfigurasi terpusat untuk project ASR (Project 3).
Semua variabel global dan parameter preprocessing disimpan di sini.
"""

# --- Konfigurasi Data ---
DATASET_NAME = "PolyAI/minds14"
LANG_SUBSET = "en-US"

# --- Konfigurasi Audio ---
TARGET_SAMPLING_RATE = 16000  # Target SR untuk LSTM, Wav2Vec2, dan Whisper
N_MFCC = 13  # Jumlah koefisien MFCC (sesuai Sesi 8)

# --- Konfigurasi Teks (dari EDA Langkah 4) ---
# Didefinisikan di sini agar bisa diimpor oleh tokenizer
VOCAB = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Ukuran vocab kita (27)
VOCAB_SIZE = len(VOCAB)

# --- Parameter Model LSTM ---
# ID 0 akan dicadangkan untuk token <blank> CTC
# Vocab asli (spasi, a-z) akan di-map ke 1-27
CTC_BLANK_TOKEN_ID = 0
# Ukuran output layer = 27 (vocab) + 1 (blank) = 28
CTC_VOCAB_SIZE = VOCAB_SIZE + 1 

# Parameter Arsitektur LSTM
LSTM_INPUT_SIZE = N_MFCC   # Input adalah 13 fitur MFCC
LSTM_HIDDEN_SIZE = 128     # Ukuran hidden state
LSTM_NUM_LAYERS = 2        # Jumlah tumpukan layer LSTM
LSTM_DROPOUT = 0.1

# Path output spesifik untuk data yang diproses MFCC
OUTPUT_DIR_MFCC = "./processed_data_mfcc" 

# --- Konfigurasi Benchmark 1: LSTM ---
OUTPUT_DIR_LSTM = "./models/lstm-mfcc-from-scratch"

# --- Konfigurasi Benchmark 2: Wave2Vec2 ---
# Menggunakan model yang sudah di fine-tune di 16kHz
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_DIR_WAV2VEC2_DATA = "./processed_data_wav2vec2" 
OUTPUT_DIR_WAV2VEC2_MODEL = "./models/wav2vec2-finetuned" 

# --- Konfigurasi Benchmark 3: Whisper ---
WHISPER_MODEL_NAME = "openai/whisper-tiny.en" # Model 'tiny' khusus English
OUTPUT_DIR_WHISPER_DATA = "./processed_data_whisper"
OUTPUT_DIR_WHISPER_MODEL = "./models/whisper-finetuned"

# --- Konfigurasi Intent Classification ---
INTENT_MODEL_NAME = "bert-base-uncased" # Model classifier standar
INTENT_MAX_LENGTH = 128 # Panjang token maks untuk intensi
OUTPUT_DIR_INTENT_DATA = "./processed_data_intent"
OUTPUT_DIR_INTENT_MODEL = "./models/intent-classifier"