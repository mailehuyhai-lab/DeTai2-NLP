"""
text_cleaner.py — Unicode normalization, emoji acronym → Unicode, whitespace normalization

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Theo PREPROCESSING_PLAN.md đã duyệt 2026-09-29

Quyết định đã duyệt:
  Q2 = A: Emoji acronym → Emoji Unicode
  Q3 = A: Giữ nguyên punctuation spacing (pre-tokenized)
  Q4 = C: Tạo 2 phiên bản (text_clean giữ case + text_lower)
"""

import re
import unicodedata


# === Bảng chuyển đổi emoji acronym → Emoji Unicode ===
# Theo PREPROCESSING_PLAN.md mục 4.3 (đã duyệt Q2=A)
#
# THỨ TỰ QUAN TRỌNG: Acronym dài trước, ngắn sau, để tránh xung đột.
# Ví dụ: colonsmilesmile trước colonsmile, colonsadcolon trước colonsad,
#         colondoublesurprise trước colonsurprise.
EMOJI_ACRONYM_MAP = [
    # --- Acronym dài trước ---
    ("colonsmilesmile", "\U0001F604"),       # 😄 Cười to
    ("colondoublesurprise", "\U0001F621"),   # 😡 Tức giận
    ("colonlovelove", "\U0001F970"),          # 🥰 Yêu thương
    ("colonsadcolon", "\U0001F622"),          # 😢 Khóc
    ("colonsmallsmile", "\U0001F60B"),        # 😋 Lè lưỡi
    ("colonbigsmile", "\U0001F606"),          # 😆 Cười lớn
    ("colonsurprise", "\U0001F62E"),          # 😮 Ngạc nhiên
    ("coloncontemn", "\U0001F60F"),           # 😏 Khinh / dễ thương
    ("colonsmile", "\U0001F642"),             # 🙂 Cười
    ("coloncolon", ">>"),                     # Giữ nguyên (không phải emoji)
    ("colonlove", "❤️"),           # ❤️ Yêu
    ("colonhihi", "\U0001F60A"),             # 😊 Vui vẻ
    ("colonsad", "\U0001F61E"),              # 😞 Buồn
    ("coloncc", "\U0001F622"),               # 😢 Khóc
    # --- Acronym ngắn / không phải emoji ---
    ("doubledot", ":"),                       # Dấu hai chấm
    ("dotdotdot", "..."),                     # Ba dấu chấm
    ("fraction", "/"),                        # Dấu gạch chéo
    ("cshrap", "c#"),                         # Ngôn ngữ lập trình
    ("vdotv", "v.v"),                         # Viết tắt "vân vân"
]


def normalize_unicode(text: str) -> str:
    """
    Bước 1: Unicode NFC normalization.
    Đảm bảo ký tự tiếng Việt nhất quán (precomposed form).
    """
    return unicodedata.normalize("NFC", text)


def replace_emoji_acronyms(text: str) -> str:
    """
    Bước 2: Chuyển emoji acronym → Emoji Unicode.
    Theo bảng chuyển đổi đã duyệt (Q2=A).
    Thứ tự: acronym dài trước, ngắn sau.
    """
    for acronym, replacement in EMOJI_ACRONYM_MAP:
        text = text.replace(acronym, replacement)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Bước 3: Chuẩn hóa khoảng trắng.
    - Nhiều khoảng trắng liên tiếp → 1 khoảng trắng.
    - Xóa khoảng trắng đầu/cuối.

    Bước 4 (Q3=A): Giữ nguyên punctuation spacing (pre-tokenized).
    → Không cần code thêm — khoảng trắng trước dấu câu ĐƯỢC giữ nguyên.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Pipeline text cleaning hoàn chỉnh.
    Áp dụng tuần tự: NFC → emoji → whitespace.
    Kết quả: text_clean (giữ case gốc).
    """
    text = normalize_unicode(text)        # Bước 1
    text = replace_emoji_acronyms(text)   # Bước 2
    text = normalize_whitespace(text)     # Bước 3 + 4
    return text


def to_lowercase(text_clean: str) -> str:
    """
    Bước 5 (Q4=C): Tạo phiên bản lowercase cho TF-IDF baseline.
    """
    return text_clean.lower()


# === Danh sách acronym để validation kiểm tra ===
ALL_ACRONYM_KEYS = [acronym for acronym, _ in EMOJI_ACRONYM_MAP]
