# backend/app/db/base.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    maternal_age = Column(Float)
    pre_pregnancy_weight = Column(Float)
    height = Column(Float)
    weight_gain = Column(Float)
    gestational_age_weeks = Column(Float) # Changed from Integer to Float as per request schema
    
    # New Fields
    conception_art = Column(Boolean, default=False)
    bmi = Column(Float, nullable=True)
    
    prediction_result = Column(String)
    confidence = Column(Float)
    model_used = Column(String)