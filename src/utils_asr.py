# src/utils_asr.py
import re

def clean_text(text):
    """Membersihkan teks transkripsi dari karakter non-alfabet."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    # Hapus semua karakter KECUALI huruf (a-z) dan spasi
    text = re.sub(r"[^a-z ]", "", text)
    # Hapus spasi ganda
    text = re.sub(r"\s+", " ", text).strip()
    return text