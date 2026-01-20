import pickle
from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt
from typing import Literal, Optional
from fastapi import FastAPI
import uvicorn

class TumorFeatures(BaseModel):
    area_mean: float = Field(..., ge=143.5, le=2501.0, description="area_mean (observed min/max)")
    area_se: float = Field(..., ge=6.802, le=542.2, description="area_se")
    area_worst: float = Field(..., ge=185.2, le=4254.0, description="area_worst")

    compactness_mean: float = Field(..., ge=0.01938, le=0.3454, description="compactness_mean")
    compactness_se: float = Field(..., ge=0.002252, le=0.1354, description="compactness_se")
    compactness_worst: float = Field(..., ge=0.02729, le=1.058, description="compactness_worst")

    concave_points_mean: float = Field(..., ge=0.0, le=0.2012, description="concave_points_mean")
    concave_points_se: float = Field(..., ge=0.0, le=0.05279, description="concave_points_se")
    concave_points_worst: float = Field(..., ge=0.0, le=0.291, description="concave_points_worst")

    concavity_mean: float = Field(..., ge=0.0, le=0.4268, description="concavity_mean")
    concavity_se: float = Field(..., ge=0.0, le=0.3960, description="concavity_se")
    concavity_worst: float = Field(..., ge=0.0, le=1.252, description="concavity_worst")

    fractal_dimension_mean: float = Field(..., ge=0.04996, le=0.09744, description="fractal_dimension_mean")
    fractal_dimension_se: float = Field(..., ge=0.000895, le=0.02984, description="fractal_dimension_se")
    fractal_dimension_worst: float = Field(..., ge=0.05504, le=0.2075, description="fractal_dimension_worst")

    perimeter_mean: float = Field(..., ge=43.79, le=188.5, description="perimeter_mean")
    perimeter_se: float = Field(..., ge=0.757, le=21.98, description="perimeter_se")
    perimeter_worst: float = Field(..., ge=50.41, le=251.2, description="perimeter_worst")

    radius_mean: float = Field(..., ge=6.981, le=28.11, description="radius_mean")
    radius_se: float = Field(..., ge=0.1115, le=2.873, description="radius_se")
    radius_worst: float = Field(..., ge=7.93, le=36.04, description="radius_worst")

    smoothness_mean: float = Field(..., ge=0.05263, le=0.1634, description="smoothness_mean")
    smoothness_se: float = Field(..., ge=0.001713, le=0.03113, description="smoothness_se")
    smoothness_worst: float = Field(..., ge=0.07117, le=0.2226, description="smoothness_worst")

    symmetry_mean: float = Field(..., ge=0.106, le=0.304, description="symmetry_mean")
    symmetry_se: float = Field(..., ge=0.007882, le=0.07895, description="symmetry_se")
    symmetry_worst: float = Field(..., ge=0.1565, le=0.6638, description="symmetry_worst")

    texture_mean: float = Field(..., ge=9.71, le=39.28, description="texture_mean")
    texture_se: float = Field(..., ge=0.3602, le=4.885, description="texture_se")
    texture_worst: float = Field(..., ge=12.02, le=49.54, description="texture_worst")


class PredictResponse(BaseModel):
    diagnosis_probability: float
    malignant: bool


app = FastAPI(title="cancer-diagnosis-detection")

with open('model.bin', 'rb') as f_in:
    saved_components = pickle.load(f_in)
    dv = saved_components["vectorizer"]
    model = saved_components['model']

def predict_single(tumor_details):
    tumor_df = dv.transform(tumor_details)
    # dtest = xgb.DMatrix(app_df, feature_names=dv.get_feature_names_out().tolist()) 
    result = model.predict(tumor_df)[0]
    return float(result)


@app.post("/predict")
def predict(tumor: TumorFeatures) -> PredictResponse:
    prob = predict_single(tumor.model_dump())

    return PredictResponse(
        diagnosis_probability=prob,
        malignant=prob >= 0.5
    )

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9696)
    except Exception as e:
        print(f"Uvicorn error: {e}")
