// frontend/src/lib/api.ts

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8008/api/v1").replace(/\/$/, '');

// Define the types for our data structures
export interface PredictionHistoryItem {
  id: number;
  timestamp: string;
  maternal_age: number;
  pre_pregnancy_weight: number;
  height: number;
  weight_gain: number;
  gestational_age_weeks: number;
  prediction_result: "Positive" | "Negative" | "Uncertain";
  confidence: number;
  model_used: string;
}

export interface PredictionFormData {
  maternalAge: string;
  prePregnancyWeight: string;
  height: string;
  weightGain: string;
  gestationalAge: string; // Now interpreted as Weeks
  conceptionArt: boolean;
  bmi?: number;
}

export interface AdvancedPredictionFormData extends PredictionFormData {
  sdgOption: string;
  algorithmOption: string;
}

// A helper function to create a detailed error message from the API response
async function createApiError(response: Response, defaultMessage: string): Promise<Error> {
  try {
    const errorData = await response.json();
    let message = defaultMessage;
    if (errorData && errorData.detail) {
      // If detail is an object/array (like FastAPI validation errors), stringify it
      if (typeof errorData.detail === 'object') {
        message = JSON.stringify(errorData.detail);
      } else {
        // Otherwise, use it as a string
        message = errorData.detail;
      }
    }
    return new Error(message);
  } catch {
    // If the response is not JSON, return the default message
    return new Error(defaultMessage);
  }
}

export async function predictCtgan(formData: PredictionFormData): Promise<PredictionHistoryItem> {
  const backendFormData = {
    "MA": Number(formData.maternalAge),
    "Wt pre": Number(formData.prePregnancyWeight),
    "Ht": Number(formData.height),
    "Wgain": Number(formData.weightGain),
    "GA weeks": Number(formData.gestationalAge),
    "Conception ART": formData.conceptionArt,
    "BMI": formData.bmi
  };

  const response = await fetch(`${API_BASE_URL}/predict-ctgan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(backendFormData),
  });
  if (!response.ok) {
    throw await createApiError(response, "CTGAN prediction failed");
  }
  return response.json();
}

export async function predictOptimal(formData: PredictionFormData): Promise<PredictionHistoryItem> {
  // Hardcoded Optimal Model: SVM SMOTE + XGBoost
  const backendFormData = {
    "MA": Number(formData.maternalAge),
    "Wt pre": Number(formData.prePregnancyWeight),
    "Ht": Number(formData.height),
    "Wgain": Number(formData.weightGain),
    "GA weeks": Number(formData.gestationalAge),
    "Conception ART": formData.conceptionArt,
    "BMI": formData.bmi,
    "sdg_option": "svm_smote",
    "algorithm_option": "XGB_model",
  };

  const response = await fetch(`${API_BASE_URL}/predict-advanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(backendFormData),
  });
  if (!response.ok) {
    throw await createApiError(response, "Optimal prediction failed");
  }
  return response.json();
}

export async function predictAdvanced(formData: AdvancedPredictionFormData): Promise<PredictionHistoryItem> {
  const backendFormData = {
    "MA": Number(formData.maternalAge),
    "Wt pre": Number(formData.prePregnancyWeight),
    "Ht": Number(formData.height),
    "Wgain": Number(formData.weightGain),
    "GA weeks": Number(formData.gestationalAge),
    "Conception ART": formData.conceptionArt,
    "BMI": formData.bmi,
    "sdg_option": formData.sdgOption,
    "algorithm_option": formData.algorithmOption,
  };

  const response = await fetch(`${API_BASE_URL}/predict-advanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(backendFormData),
  });
  if (!response.ok) {
    throw await createApiError(response, "Advanced prediction failed");
  }
  return response.json();
}


export async function getHistory(): Promise<PredictionHistoryItem[]> {
  const response = await fetch(`${API_BASE_URL}/history`);
  if (!response.ok) {
    throw new Error("Failed to fetch history");
  }
  return response.json();
}

export async function deleteHistoryItem(id: number | string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/history/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Failed to delete history item with id: ${id}`);
  }
  // No content is expected on a successful delete
}
