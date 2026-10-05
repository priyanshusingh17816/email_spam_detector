import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

data = [
    ("Congratulations! You won a $1000 gift card. Click now to claim your prize!", 1),
    ("URGENT! You have won a lottery. Send your details to receive the money.", 1),
    ("Free entry in our weekly prize draw. Text WIN to 12345 now.", 1),
    ("You have been selected for a cash reward. Claim it immediately.", 1),
    ("Get cheap loans instantly. Apply now and receive money today.", 1),
    ("Limited offer! Buy now and get 90% discount.", 1),
    ("Your account has won a special bonus. Click the link to claim.", 1),
    ("Congratulations winner! You have received a free vacation package.", 1),

    ("Hi, are we still meeting for lunch today?", 0),
    ("Please find the assignment attached. Submit it by tomorrow.", 0),
    ("Can you call me when you are free?", 0),
    ("Your Amazon order has been shipped and will arrive tomorrow.", 0),
    ("Reminder: your doctor appointment is scheduled for Monday.", 0),
    ("Let's meet at 5 PM near the library.", 0),
    ("The project meeting has been moved to 10 AM.", 0),
    ("Happy birthday! Hope you have a great day.", 0),
]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

os.makedirs("model", exist_ok=True)
with open("model/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained successfully.")
