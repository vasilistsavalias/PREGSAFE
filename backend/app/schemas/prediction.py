from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PatientData(BaseModel):
    """
    Defines the data schema for patient information provided to the API.
    """
    ma: float = Field(..., alias="MA", description="Maternal Age")
    wt_pre: float = Field(..., alias="Wt pre", description="Weight Pre-Pregnancy (kg)")
    ht: float = Field(..., alias="Ht", description="Height (cm)")
    wgain: float = Field(..., alias="Wgain", description="Weight Gain during pregnancy (kg)")
    # UPDATED: Input is now Weeks, not Days
    ga_weeks: float = Field(..., alias="GA weeks", description="Gestational Age in weeks")
    # NEW: Assisted Reproductive Technology
    conception_art: bool = Field(default=False, alias="Conception ART", description="Conception via ART (IVF etc)")
    # NEW: Pre-calculated BMI (optional, can be computed backend if missing)
    bmi: Optional[float] = Field(default=None, alias="BMI", description="Body Mass Index")


class AdvancedPredictionRequest(PatientData):
    """
    Extends PatientData to include model selection options for the R service.
    """
    sdg_option: str = Field(..., description="The Synthetic Data Generation model to use (e.g., 'smote')")
    algorithm_option: str = Field(..., description="The ML algorithm to use (e.g., 'RF_model')")


class Prediction(BaseModel):
    """
    Pydantic schema for returning a prediction record.
    This is used as the `response_model` in API endpoints.
    """
    id: int
    timestamp: datetime
    maternal_age: float
    pre_pregnancy_weight: float
    height: float
    weight_gain: Optional[float] = None # Allow None/null values
    gestational_age_weeks: float 
    prediction_result: str
    confidence: float
    model_used: str

    class Config:
        from_attributes = True
