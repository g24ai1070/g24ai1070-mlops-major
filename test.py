import os, joblib
from utils import get_splits
from sklearn.metrics import accuracy_score

MODEL_PATH = "artifacts/savedmodel.pth"

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run train.py first")
    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = get_splits()
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
