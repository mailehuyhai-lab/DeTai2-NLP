"""
tfidf.py — TF-IDF vectorizer setup for baseline models

Đề tài #2: Phân tích cảm xúc và chủ đề từ phản hồi người dùng
Baseline: TF-IDF + Logistic Regression + Linear SVM

TF-IDF chỉ fit trên TRAIN. Dev/Test chỉ transform.
Sử dụng text_lower (đã duyệt Q4=C).
"""

from sklearn.feature_extraction.text import TfidfVectorizer


# Cấu hình TF-IDF baseline cho Vietnamese short feedback
TFIDF_CONFIG = {
    "ngram_range": (1, 2),      # Unigrams + bigrams
    "min_df": 2,                # Loại bỏ từ xuất hiện < 2 lần
    "max_df": 0.95,             # Loại bỏ từ xuất hiện > 95% documents
    "sublinear_tf": True,       # Dùng 1 + log(tf) thay vì tf thuần
    "max_features": 50000,      # Giới hạn vocab để tránh quá lớn
    "strip_accents": None,      # Không strip accents (tiếng Việt cần dấu)
    "lowercase": False,         # text_lower đã lowercase sẵn
}


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Tạo TF-IDF vectorizer với cấu hình baseline.

    Returns:
        TfidfVectorizer chưa fit.
    """
    return TfidfVectorizer(**TFIDF_CONFIG)
