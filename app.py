from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "spam_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    email = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip()

        if email:
            prediction = model.predict([email])[0]
            result = "SPAM" if prediction == 1 else "NOT SPAM"

    return render_template("index.html", result=result, email=email)

if __name__ == "__main__":
    app.run(debug=True)
