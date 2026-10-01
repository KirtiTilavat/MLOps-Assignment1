"""Request / response schemas for the prediction API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """The 13 clinical attributes of the UCI Heart Disease (Cleveland) dataset."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 54,
                "sex": 1,
                "cp": 4,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 2,
                "thalach": 140,
                "exang": 1,
                "oldpeak": 1.6,
                "slope": 2,
                "ca": 1,
                "thal": 7,
            }
        },
    )

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Major vessels coloured by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(
        ..., description="Thallium test: 3 normal, 6 fixed defect, 7 reversible defect"
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_level: Literal["low", "moderate", "high"]
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
