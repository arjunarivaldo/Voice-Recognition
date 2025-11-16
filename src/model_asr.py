# src/model_asr.py
import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
# CausalLMOutput adalah output standar yang berisi "loss" dan "logits"

# --- 1. Config (Mirip BertConfig) ---
class LstmCtcConfig(PretrainedConfig):
    """
    Konfigurasi untuk model LSTM-CTC from-scratch.
    """
    model_type = "lstm-ctc-from-scratch"
    
    def __init__(
        self,
        input_size=13,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        vocab_size=28, # (len(vocab) + 1 blank)
        ctc_blank_token_id=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.ctc_blank_token_id = ctc_blank_token_id

# --- 2. Model (Mirip BertSumClassifier) ---
class LstmCtcForAsr(PreTrainedModel):
    """
    Model LSTM-CTC from-scratch.
    """
    # Beri tahu Hugging Face nama config-nya
    config_class = LstmCtcConfig
    
    def __init__(self, config: LstmCtcConfig):
        super().__init__(config)
        
        # --- Arsitektur Model ---
        self.lstm = nn.LSTM(
            input_size=config.input_size,       # 13 (dari MFCC)
            hidden_size=config.hidden_size,     # 128 (dari config)
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=True,                 # Penting untuk konteks
            batch_first=True                    # Input kita (B, T, F)
        )
        
        # Layer Linear (Classifier)
        # Output = hidden_size * 2 (karena bidirectional)
        self.classifier = nn.Linear(
            config.hidden_size * 2, 
            config.vocab_size                   # 28 (dari config)
        )
        
        # Layer LogSoftmax (Wajib untuk CTCLoss PyTorch)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        # Fungsi CTC Loss
        # zero_infinity=True: Mencegah error jika input terlalu pendek
        self.ctc_loss = nn.CTCLoss(
            blank=config.ctc_blank_token_id, 
            zero_infinity=True
        )

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None, # (Akan disuplai Collator)
        label_lengths: Optional[torch.Tensor] = None, # (Akan disuplai Collator)
        attention_mask: Optional[torch.Tensor] = None, # (Trainer membuatnya, kita bisa pakai)
        **kwargs
    ):
        # 1. Lewatkan ke LSTM
        # output shape: (batch_size, time_steps, hidden_size * 2)
        lstm_output, _ = self.lstm(input_features)
        
        # 2. Lewatkan ke Classifier
        # logits shape: (batch_size, time_steps, vocab_size)
        logits = self.classifier(lstm_output)
        
        # 3. Lewatkan ke LogSoftmax (Wajib untuk CTC)
        # log_probs shape: (batch_size, time_steps, vocab_size)
        log_probs = self.log_softmax(logits)
        
        loss = None
        if labels is not None:
            # --- Jika 'labels' diberikan (mode training), hitung loss ---
            
            # Kita perlu panjang (unpadded) dari input (MFCC)
            if input_lengths is None and attention_mask is not None:
                # Hitung panjang input dari attention_mask (jika collator memberikannya)
                input_lengths = attention_mask.sum(-1).long()
            elif input_lengths is None:
                # Jika tidak ada, asumsikan semua input penuh (fallback)
                input_lengths = torch.full(
                    (log_probs.size(0),), log_probs.size(1), 
                    dtype=torch.long, device=log_probs.device
                )

            # Kita perlu panjang (unpadded) dari label
            if label_lengths is None:
                # Hitung panjang label dari labels (abaikan -100)
                label_lengths = (labels != -100).sum(-1).long()

            # --- Perhitungan CTC Loss ---
            # CTCLoss di PyTorch butuh input (Time, Batch, Features)
            # Jadi kita transpose log_probs
            log_probs_t = log_probs.transpose(0, 1) # (T, B, V)

            loss = self.ctc_loss(
                log_probs_t,
                labels,
                input_lengths,
                label_lengths
            )

        # Kembalikan output yang kompatibel dengan Trainer
        # Mirip dengan SequenceClassifierOutput di Project 2 Anda
        return CausalLMOutput(
            loss=loss,
            logits=logits, # (Kita kembalikan logits mentah, bukan log_probs)
        )