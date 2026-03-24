from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# =====================
# Load Model
# =====================

with open("medical_healthcare_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🏥 Healthcare Prediction API Running"}


# =====================
# Input Schema
# =====================

class InputData(BaseModel):
    Sex: float
    BMI: float
    Smoking: float
    AlcoholDrinking: float
    HeartDisease: float
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: float
    Diabetic: float
    AgeCategory: float
    PhysicalActivity: float
    GenHealth: float
    SleepTime: float
    Asthma: float
    KidneyDisease: float


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
            0: "Arthritis",
            1: "Asthma",
            2: "Cancer",
            3: "Diabetes",
            4: "Healthy",
            5: "Hypertension",
            6: "Obesity"
        }

        pred = int(prediction[0])

        return {
            "prediction": pred,
            "disease": class_mapping.get(pred, "Unknown"),
            "probability": float(probability[0].max())
        }

    except Exception as e:
        return {"error": str(e)}
