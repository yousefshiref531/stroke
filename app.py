from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# =====================
# Load Model
# =====================

with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🧠 Stroke Prediction API Running"}


# =====================
# Input Schema
# ⚠️ عدلها لو الأعمدة مختلفة
# =====================

class InputData(BaseModel):
    age: float
    hypertension: float
    heart_disease: float
    avg_glucose_level: float
    bmi: float


# =====================
# Predict
# =====================

@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])

        prediction = model.predict(df)
        probability = model.predict_proba(df)

        # mapping
        class_mapping = {
            0: "No Stroke",
            1: "Stroke"
        }

        pred = int(prediction[0])

        return {
            "prediction": pred,
            "disease": class_mapping[pred],
            "probability": float(probability[0][1])
        }

    except Exception as e:
        return {"error": str(e)}
