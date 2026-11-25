import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from predict import load_model, predict_pair


ROOT = Path(__file__).resolve().parents[1]


class SmokeTest(unittest.TestCase):
    def test_training_saved_model_and_app(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = []
            for topic in ["python", "java", "guitar", "cooking", "swimming",
                          "drawing", "algebra", "chess", "singing", "running"]:
                for number in range(10):
                    rows.append([f"how to learn {topic} lesson {number}",
                                 f"how to study {topic} lesson {number}", 1])
                    rows.append([f"how to start {topic} course {number}",
                                 f"how much does {topic} equipment {number} cost", 0])
            data = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
            data.to_csv(folder / "sample.csv", index=False)
            command = [
                sys.executable, str(ROOT / "train.py"),
                "--data", str(folder / "sample.csv"), "--sample-size", "0",
                "--max-features", "100", "--trees", "5", "--output", str(folder)
            ]
            try:
                import xgboost
                command.append("--with-xgboost")
            except ImportError:
                pass
            subprocess.run(command, check=True, timeout=180)
            model_path = folder / "artifacts" / "model.pkl"
            artifacts = load_model(model_path)
            first = predict_pair("learn python", "study python", artifacts)
            second = predict_pair("study python", "learn python", artifacts)
            self.assertEqual(first, second)
            self.assertTrue(0 <= first[1] <= 1)
            metrics = json.loads((folder / "reports" / "test_metrics.json").read_text())
            self.assertIn("f1", metrics)
            self.assertTrue((folder / "reports" / "confusion_matrix.png").exists())
            with patch("predict.MODEL_PATH", folder / "missing.pkl"):
                app = AppTest.from_file(str(ROOT / "app.py")).run()
                self.assertFalse(app.exception)
                self.assertIn("Train the model first", app.info[0].value)
            with patch("predict.MODEL_PATH", model_path):
                with patch("predict.load_model", lambda: artifacts):
                    app = AppTest.from_file(str(ROOT / "app.py")).run()
                    self.assertFalse(app.exception)
                    app.button[0].click().run()
                    self.assertTrue(app.warning)
                    app.text_area[0].input("How can I learn Python?")
                    app.text_area[1].input("How do I study Python?")
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(len(app.metric), 1)


if __name__ == "__main__":
    unittest.main()
