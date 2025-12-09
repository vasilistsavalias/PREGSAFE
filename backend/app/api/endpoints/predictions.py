# backend/app/api/endpoints/predictions.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.schemas.prediction import PatientData, Prediction as PredictionSchema, AdvancedPredictionRequest
from app.services import prediction_service

router = APIRouter()

@router.post("/predict-ctgan", response_model=PredictionSchema)
async def predict_ctgan_endpoint(data: PatientData, db: Session = Depends(get_db)):
    return await prediction_service.predict_with_ctgan(data, db)

@router.post("/predict-advanced", response_model=PredictionSchema)
async def predict_advanced_endpoint(data: AdvancedPredictionRequest, db: Session = Depends(get_db)):
    """
    Endpoint to run predictions using the external R microservice.
    Accepts advanced model selection options.
    """
    return await prediction_service.predict_with_r_models(data, db)


@router.get("/history", response_model=List[PredictionSchema])
def get_history_endpoint(db: Session = Depends(get_db)):
    return prediction_service.get_history(db)

@router.delete("/history/{prediction_id}", status_code=200)
def delete_history_item(prediction_id: int, db: Session = Depends(get_db)):
    return prediction_service.delete_prediction_by_id(db, prediction_id)
