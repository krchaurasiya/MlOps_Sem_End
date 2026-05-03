from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

data = joblib.load("model/rank_model.pkl")

model = data["model"]
tfidf = data["tfidf"]
le = data["label_encoder"]

@app.post("/predict")
def predict(input_data: dict):
    
    # Encode category
    category_encoded = le.transform([input_data["category"]])[0]

    # TF-IDF transform
    text_features = tfidf.transform([input_data["text"]]).toarray()

    # Combine all features
    final_input = np.hstack([
        [[input_data["rating"], input_data["review_count"], category_encoded]],
        text_features
    ])

    prediction = model.predict(final_input)

    return {"prediction": prediction.tolist()}