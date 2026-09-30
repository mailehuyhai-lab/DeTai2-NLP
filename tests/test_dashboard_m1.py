"""Lightweight regression tests for the MỨC 1 dashboard additions."""
from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

import pandas as pd

from src.dashboard import inference, loader
from src.dashboard.views import batch, error_low_confidence


class CsvUploadTests(unittest.TestCase):
    def test_read_uploaded_csv_and_default_text_column(self):
        class Uploaded:
            def getvalue(self):
                return "text\nGiảng viên dạy rất dễ hiểu\n".encode("utf-8")

        frame = batch.read_uploaded_csv(Uploaded())
        self.assertEqual(list(frame.columns), ["text"])
        self.assertEqual(batch.choose_default_text_column(frame.columns), "text")

    def test_empty_csv_is_rejected(self):
        class Uploaded:
            def getvalue(self):
                return b""

        with self.assertRaisesRegex(ValueError, "rỗng"):
            batch.read_uploaded_csv(Uploaded())

    def test_header_only_csv_is_rejected(self):
        class Uploaded:
            def getvalue(self):
                return b"text\n"

        with self.assertRaisesRegex(ValueError, "header"):
            batch.read_uploaded_csv(Uploaded())

    def test_no_named_text_column_requires_explicit_selection(self):
        self.assertIsNone(batch.choose_default_text_column(["note1", "note2"]))
        self.assertEqual(batch.choose_default_text_column(["feedback"]), "feedback")


class BatchResultTests(unittest.TestCase):
    def test_batch_preserves_columns_and_reports_invalid_rows(self):
        frame = pd.DataFrame({
            "id": [1, 2, 3],
            "text": ["Giảng viên dạy rất dễ hiểu", "", None],
        })

        def fake_predict(texts):
            calls = []
            for i, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    calls.append({"row_number": i, "status": "invalid_input", "error": "empty", "result": None})
                else:
                    calls.append({
                        "row_number": i,
                        "status": "success",
                        "error": "",
                        "result": {
                            "text_clean": text,
                            "truncated": False,
                            "sentiment": {"id": 2, "label": "Positive", "confidence": 0.9, "probs": {}},
                            "topic": {"id": 0, "label": "Lecturer", "confidence": 0.8, "probs": {}},
                        },
                    })
            return calls

        result = batch.build_batch_results(frame, "text", predictor=fake_predict)
        self.assertEqual(list(result.columns[:2]), ["id", "text"])
        self.assertIn("predicted_sentiment", result.columns)
        self.assertIn("sentiment_confidence", result.columns)
        self.assertIn("predicted_topic", result.columns)
        self.assertIn("topic_confidence", result.columns)
        self.assertEqual(result.loc[0, "predicted_sentiment"], "Positive")
        self.assertTrue(pd.isna(result.loc[1, "predicted_sentiment"]))
        self.assertEqual(result.loc[1, "prediction_status"], "invalid_input")

    def test_export_is_utf8_sig_and_reloadable(self):
        frame = pd.DataFrame({"text": ["Giảng viên dạy rất dễ hiểu 😄"]})
        payload = batch.dataframe_csv_bytes(frame)
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        loaded = pd.read_csv(BytesIO(payload))
        self.assertEqual(loaded.loc[0, "text"], frame.loc[0, "text"])

    def test_predict_feedback_batch_delegates_and_marks_input_errors(self):
        with patch.object(inference, "predict_feedback", side_effect=[
            {"sentiment": {"label": "Positive"}, "topic": {"label": "Lecturer"}},
            ValueError("rỗng"),
        ]) as mocked:
            records = inference.predict_feedback_batch(["ok", " "])
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(records[0]["status"], "success")
        self.assertEqual(records[1]["status"], "invalid_input")


class ErrorFilterTests(unittest.TestCase):
    def test_errors_only_uses_correct_column(self):
        frame = pd.DataFrame({
            "id": ["a", "b"],
            "text": ["x", "y"],
            "true_label": [1, 0],
            "predicted_label": [0, 0],
            "correct": [0, 1],
            "true_label_name": ["Neutral", "Negative"],
            "predicted_label_name": ["Negative", "Negative"],
        })
        errors = error_low_confidence.filter_test_errors(frame)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors.loc[0, "id"], "a")

    def test_static_predictions_do_not_claim_confidence(self):
        frame = loader.load_csv(loader.tf_pred_path("sentiment"))
        self.assertFalse(loader.transformer_prediction_has_confidence(frame))

    def test_batch_low_confidence_uses_real_softmax_column(self):
        frame = pd.DataFrame({
            "text": ["a", "b", "c"],
            "predicted_sentiment": ["Positive", "Negative", "Neutral"],
            "sentiment_confidence": [0.4, 0.7, None],
            "predicted_topic": ["Lecturer", "Facility", "Others"],
            "topic_confidence": [0.9, 0.5, 0.2],
            "prediction_status": ["success", "success", "invalid_input"],
        })
        low = error_low_confidence.filter_batch_results(frame, "sentiment", 0.6, "Low confidence")
        self.assertEqual(low["text"].tolist(), ["a"])
        errors = error_low_confidence.filter_batch_results(frame, "sentiment", 0.6, "Errors only")
        self.assertTrue(errors.empty)


if __name__ == "__main__":
    unittest.main()
