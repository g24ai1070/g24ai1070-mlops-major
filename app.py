from flask import Flask, request, render_template
import joblib, os
import numpy as np
from PIL import Image

app = Flask(__name__)
MODEL_PATH = "artifacts/savedmodel.pth"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files['image']
        img = Image.open(file.stream).convert('L').resize((64, 64))
        flat = np.array(img).reshape(1, -1)
        model = joblib.load(MODEL_PATH)
        pred = model.predict(flat)[0]
        result = f"Predicted class: {pred}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
