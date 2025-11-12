from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

def load_olivetti_flat():
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.images
    y = data.target
    n, h, w = X.shape
    X_flat = X.reshape(n, h*w)
    return X_flat, y

def get_splits(test_size=0.3, random_state=42):
    X, y = load_olivetti_flat()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
