# test_model.py
import joblib
import sys
import os

MODEL_PATH = os.path.join("..", "models", "ddi_baseline_model.joblib")  # adjust if needed

def load_model(path):
    m = joblib.load(path)
    print(f"Loaded model from: {path}")
    return m

def get_prediction_and_confidence(model, text):
    # Ensure text is a list for sklearn
    X = [text]

    # Try predict_proba first
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # try to find index for class label 1 (interaction). fall back to last index
        try:
            idx_1 = list(model.classes_).index(1)
        except Exception:
            idx_1 = len(probs) - 1
        confidence = float(probs[idx_1])
        pred = int(model.predict(X)[0])
        return pred, confidence, probs, list(model.classes_)
    # If model is a pipeline that wraps an inner estimator without predict_proba, try to find it
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "predict_proba"):
                probs = step.predict_proba(X)[0]
                try:
                    idx_1 = list(step.classes_).index(1)
                except Exception:
                    idx_1 = len(probs)-1
                confidence = float(probs[idx_1])
                pred = int(model.predict(X)[0])
                return pred, confidence, probs, list(step.classes_)
    # fallback: try decision_function -> convert to pseudo-prob via sigmoid
    if hasattr(model, "decision_function"):
        import math
        score = model.decision_function(X)[0]
        # sigmoid
        confidence = 1 / (1 + math.exp(-score))
        pred = int(model.predict(X)[0])
        return pred, confidence, None, None

    # final fallback: only predict available
    pred = int(model.predict(X)[0])
    return pred, None, None, None

def pretty_print(pred, confidence, probs, classes, text):
    print("=== INPUT ===")
    print(text)
    print("=== RESULT ===")
    if pred == 1:
        print("Prediction: INTERACTION (1)")
    else:
        print("Prediction: NO INTERACTION (0)")
    if confidence is not None:
        print(f"Confidence (interaction=1): {confidence:.4f} ({confidence*100:.1f}%)")
    else:
        print("Confidence: not available for this model (no predict_proba/decision_function).")
    if probs is not None and classes is not None:
        print("Detailed class probabilities:")
        for c, p in zip(classes, probs):
            print(f"  class {c} -> {p:.4f}")

if __name__ == "__main__":
    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Update MODEL_PATH in the script.")
        sys.exit(1)

    model = load_model(MODEL_PATH)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to classify (e.g. 'ketoconazole with atorvastatin'): ").strip()
        if not text:
            print("No input provided. Exiting.")
            sys.exit(1)

    pred, confidence, probs, classes = get_prediction_and_confidence(model, text)
    pretty_print(pred, confidence, probs, classes, text)
