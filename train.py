import os, joblib
from sklearn.tree import DecisionTreeClassifier
from utils import get_splits

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "savedmodel.pth")

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = get_splits()
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
