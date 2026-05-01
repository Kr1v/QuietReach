"""
model/classifier.py — load and run the threat classifier

Tries TFLite first, falls back to sklearn pickle.
Both return a float confidence score in [0.0, 1.0].

Inference used to be slow enough to miss windows (~180ms on CPU).
Fixed by pre-allocating the TFLite interpreter tensors once at load time
instead of re-allocating on every call. Now consistently under 15ms.
# that was a painful bug to track down
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from audio.processor import FEATURE_DIM

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    """
    Structural protocol — both TFLite and sklearn wrappers satisfy this.
    Lets threshold.py and tests reference a single type without caring
    which backend is loaded.
    """
    def predict(self, feature_vec: np.ndarray) -> float:
        ...

    @property
    def backend(self) -> str:
        ...


class TFLiteClassifier:
    """
    Wraps a TFLite flatbuffer. Allocates tensors once at init.
    Input: (1, FEATURE_DIM) float32
    Output: (1, 1) float32 sigmoid probability
    """

    def __init__(self, model_path: str) -> None:
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._interpreter.allocate_tensors()
        except ImportError:
            raise RuntimeError("TensorFlow not installed — cannot load TFLite model")

        details = self._interpreter.get_input_details()
        self._input_index = details[0]["index"]
        self._input_shape = details[0]["shape"]  # expect [1, FEATURE_DIM]

        out_details = self._interpreter.get_output_details()
        self._output_index = out_details[0]["index"]

        expected_input = (1, FEATURE_DIM)
        if tuple(self._input_shape) != expected_input:
            raise ValueError(
                f"TFLite model expects input shape {tuple(self._input_shape)}, "
                f"but feature vector is {expected_input}. "
                "Did you retrain with a different FEATURE_DIM?"
            )

        logger.info(f"TFLite classifier loaded from {model_path}")

    def predict(self, feature_vec: np.ndarray) -> float:
        """
        Run inference. feature_vec must be shape (FEATURE_DIM,) float32.
        Returns confidence score 0.0–1.0.
        """
        inp = feature_vec.reshape(1, FEATURE_DIM).astype(np.float32)
        self._interpreter.set_tensor(self._input_index, inp)

        t0 = time.monotonic()
        self._interpreter.invoke()
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"TFLite invoke: {elapsed_ms:.1f}ms")

        output = self._interpreter.get_tensor(self._output_index)
        return float(output[0][0])

    @property
    def backend(self) -> str:
        return "tflite"


class SKLearnClassifier:
    """
    Wraps a scikit-learn classifier pickle (GradientBoosting or similar).
    Expects the model to have a predict_proba method.
    """

    def __init__(self, model_path: str) -> None:
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        if not hasattr(self._model, "predict_proba"):
            raise ValueError(
                f"sklearn model at {model_path} has no predict_proba. "
                "Needs a probabilistic classifier (GradientBoosting, RandomForest, etc.)"
            )

        logger.info(f"sklearn classifier loaded from {model_path}")

    def predict(self, feature_vec: np.ndarray) -> float:
        """
        Run inference. feature_vec shape (FEATURE_DIM,).
        Returns class-1 (threat) probability.
        """
        inp = feature_vec.reshape(1, -1).astype(np.float32)
        t0 = time.monotonic()
        proba = self._model.predict_proba(inp)
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"sklearn predict_proba: {elapsed_ms:.1f}ms")
        return float(proba[0][1])  # index 1 = threat class

    @property
    def backend(self) -> str:
        return "sklearn"


def load_classifier(
    tflite_path: str,
    sklearn_path: str,
) -> Classifier:
    """
    Try TFLite first. If the file doesn't exist or TF isn't installed,
    fall back to sklearn pickle. If neither exists, raise clearly.

    This is the only function main.py should call from this module.
    """
    tflite_file = Path(tflite_path)
    sklearn_file = Path(sklearn_path)

    if tflite_file.exists():
        try:
            clf = TFLiteClassifier(str(tflite_file))
            logger.info("Using TFLite backend")
            return clf
        except (RuntimeError, ValueError, Exception) as e:
            logger.warning(f"TFLite load failed ({e}), trying sklearn fallback")

    if sklearn_file.exists():
        try:
            clf = SKLearnClassifier(str(sklearn_file))
            logger.info("Using sklearn backend")
            return clf
        except Exception as e:
            logger.error(f"sklearn load also failed: {e}")
            raise

    raise FileNotFoundError(
        f"No model found at '{tflite_path}' or '{sklearn_path}'. "
        "Run `python -m model.trainer` to train one first."
    )


class DummyClassifier:
    """
    Returns a fixed score. Useful in tests and during UI development
    before the model is trained.
    """

    def __init__(self, fixed_score: float = 0.1) -> None:
        self._score = fixed_score

    def predict(self, feature_vec: np.ndarray) -> float:  # noqa: ARG002
        return self._score

    @property
    def backend(self) -> str:
        return "dummy"
