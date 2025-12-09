# backend/app/services/prediction_service.py
import pandas as pd
import joblib
from fastapi import HTTPException
from sqlalchemy.orm import Session
import logging
from datetime import datetime, timezone
import httpx
import os

from app.db.base import Prediction
from app.schemas.prediction import PatientData, AdvancedPredictionRequest
from app.core.config import PREDICTOR_PATH, SMOTE_PREDICTOR_PATH

logger = logging.getLogger(__name__)

# Get R service URL and API key from environment variables
R_SERVICE_URL = os.environ.get("R_SERVICE_URL", "http://localhost:8000/predict") 
R_SERVICE_API_KEY = os.environ.get("R_SERVICE_API_KEY", "a-secure-secret-key")


# --- Prediction Functions ---

async def predict_with_r_models(data: AdvancedPredictionRequest, db: Session) -> Prediction:
    """
    Makes a resilient prediction request to the R microservice, handles retries,
    and saves the result to the database.
    """
    logger.info(f"--- Starting Advanced Prediction via R Service (Model: {data.sdg_option} + {data.algorithm_option}) ---")

    # --- 1. Proactive Input Validation ---
    # Validate all incoming data (Ranges updated for medical realism)
    # Limits derived from Training Data Analysis:
    # MA: 15 - 50
    # Wt pre: 35 - 150
    # BMI: 14.5 - 53.5
    # Wgain: -28 - 31
    # GA days: 140 - 174 (20 - 25 weeks)

    if not (15 <= data.ma <= 50):
         raise HTTPException(status_code=400, detail=f"Maternal Age {data.ma} is outside the supported range (15-50 years).")
         
    if not (35 <= data.wt_pre <= 150):
         raise HTTPException(status_code=400, detail=f"Pre-pregnancy Weight {data.wt_pre}kg is outside the supported range (35-150 kg).")

    if data.bmi and not (14.5 <= data.bmi <= 54):
         # Soft warning or strict? Let's be strict for safety as requested.
         raise HTTPException(status_code=400, detail=f"BMI {data.bmi} is outside the supported range (14.5-54).")
         
    if not (-28 <= data.wgain <= 31):
         raise HTTPException(status_code=400, detail=f"Weight Gain {data.wgain}kg is outside the supported range (-28 to 31 kg).")
         
    if not (1 <= data.ga_weeks <= 42):
        raise HTTPException(status_code=400, detail=f"Gestational Age {data.ga_weeks} weeks is outside the supported range (1-42 weeks).")

    # Prepare the payload for the R service
    # R service expects: maternal_age, Wt_pre, height, Wgain, GA_weeks, Conception_ART, bmi
    payload = {
        "maternal_age": data.ma,
        "Wt_pre": data.wt_pre,
        "height": data.ht,
        "Wgain": data.wgain,
        "GA_weeks": data.ga_weeks,
        "Conception_ART": 1 if data.conception_art else 0,
        "bmi": data.bmi if data.bmi else 0, # Send 0 if None, R handles it
        "SDG_option": data.sdg_option,
        "algorithm_option": data.algorithm_option,
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": R_SERVICE_API_KEY,
        # Cloudflare Bypass: Use a standard browser User-Agent to avoid the "Just a moment..." challenge
        # This is required because we are forced to use the Public URL (Free Tier limitation)
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # Use httpx with a timeout and retry mechanism
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(transport=transport, timeout=120.0) as client:
        try:
            response = await client.post(R_SERVICE_URL, json=payload, headers=headers)
            response.raise_for_status() 
            r_result = response.json()

        except httpx.RequestError as e:
            logger.critical(f"Request to R service failed: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"Could not communicate with the R prediction service.")
        except httpx.HTTPStatusError as e:
            logger.error(f"R service returned an error: {e.response.status_code} {e.response.text}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"The R prediction service returned an error.")

    # Process the result from the R service
    try:
        # Extract confidence
        # R returns: "Probability of GDM positive" which might be a float or a list [float]
        confidence_val = r_result.get("Probability of GDM positive", 0)
        
        if isinstance(confidence_val, list):
            # Take the first element if it's a list
            if len(confidence_val) > 0:
                confidence_val = confidence_val[0]
            else:
                confidence_val = 0

        confidence = float(confidence_val) * 100
        
        # Normalize R service output with "Uncertain" buffer zone
        if confidence > 55:
            result_text = "Positive"
        elif confidence < 45:
            result_text = "Negative"
        else:
            result_text = "Uncertain"
            
        model_used = f"{data.sdg_option} + {data.algorithm_option}"

        db_prediction = Prediction(
            timestamp=datetime.now(timezone.utc),
            maternal_age=data.ma,
            pre_pregnancy_weight=data.wt_pre,
            height=data.ht,
            weight_gain=data.wgain,
            gestational_age_weeks=data.ga_weeks,
            conception_art=data.conception_art,
            bmi=data.bmi,
            prediction_result=result_text,
            confidence=confidence,
            model_used=model_used
        )

        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        logger.info(f"--- R Service Prediction Successful. Saved as ID: {db_prediction.id} ---")
        return db_prediction

    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error processing R service response: {r_result}. Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail="Received malformed data from prediction service.")
    except Exception as e:
        logger.critical(f"An unhandled exception occurred after R service call: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An internal error occurred while saving the prediction.")


async def predict_with_ctgan(data: PatientData, db: Session) -> Prediction:
    """
    Legacy/Optimal Path: Now a proxy for the Best R Model (SMOTE + Naive Bayes).
    This ensures 'Optimal' and 'Advanced -> SMOTE + Naive Bayes' are mathematically identical.
    """
    logger.info("--- Optimal Request (Redirecting to R Service: SMOTE + Naive Bayes) ---")
    
    # Construct Advanced Request with Hardcoded Optimal Params
    advanced_data = AdvancedPredictionRequest(
        **data.dict(by_alias=True),
        sdg_option="smote",
        algorithm_option="NB_model"
    )
    
    return await predict_with_r_models(advanced_data, db)


# --- History Functions ---

def get_history(db: Session):
    """Fetches all prediction records from the database."""
    return db.query(Prediction).order_by(Prediction.timestamp.desc()).all()

def delete_prediction_by_id(db: Session, prediction_id: int):
    """Deletes a prediction record from the database by its ID."""
    prediction_to_delete = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction_to_delete:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    db.delete(prediction_to_delete)
    db.commit()
    return {"status": "success", "message": f"Prediction {prediction_id} deleted"}