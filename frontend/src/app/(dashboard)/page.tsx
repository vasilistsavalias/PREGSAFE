"use client";

import { useState, useMemo, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import PredictionHistoryCard from "@/components/PredictionHistoryCard";
import { predictCtgan, predictAdvanced, predictOptimal, type PredictionFormData, type AdvancedPredictionFormData, type PredictionHistoryItem } from "@/lib/api";
import { usePredictionHistory } from "@/context/PredictionContext";
import { DropdownMenu, DropdownMenuContent, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuTrigger } from "@/components/ui/dropdown";
import { ChevronDown, Loader2, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";

const SDG_OPTIONS = ['Original', 'borderline_smote', 'enn_smote', 'kmeans_smote', 'smote', 'svm_smote', 'tomek_smote', 'CTGAN'];
const ALGORITHM_OPTIONS = ['XGB_model', 'RF_model', 'KNN_model', 'SVM_model', 'NB_model'];
const OPTIMAL_SDG = 'svm_smote';
const OPTIMAL_ALGORITHM = 'XGB_model';

// Helper to check range and return warning if invalid
const checkRange = (value: string, min: number, max: number, label: string): string | null => {
  if (!value) return null;
  const num = parseFloat(value);
  if (isNaN(num)) return null; // Let the submit validation handle NaN
  if (num < min || num > max) return `${label} should be between ${min} and ${max}`;
  return null;
};

export default function Home() {
  // Form input state
  const [maternalAge, setMaternalAge] = useState("");
  const [prePregnancyWeight, setPrePregnancyWeight] = useState("");
  const [height, setHeight] = useState("");
  const [weightGain, setWeightGain] = useState("");
  const [gestationalAge, setGestationalAge] = useState("");
  const [conceptionArt, setConceptionArt] = useState(false);
  const [calculatedBmi, setCalculatedBmi] = useState<number | null>(null);

  // UI and Prediction State
  const { history, addPrediction } = usePredictionHistory();
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSeconds, setLoadingSeconds] = useState(0); // Track loading duration
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);

  // Advanced model selection state
  const [sdgOption, setSdgOption] = useState(OPTIMAL_SDG);
  const [algorithmOption, setAlgorithmOption] = useState(OPTIMAL_ALGORITHM);

  // Timer for smart loading message
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isLoading) {
      setLoadingSeconds(0);
      interval = setInterval(() => {
        setLoadingSeconds((prev) => prev + 1);
      }, 1000);
    } else {
      setLoadingSeconds(0);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  // Validation Warnings
  const warnings = useMemo(() => {
    const list = [];
    // Ranges from backend validation
    const wMa = checkRange(maternalAge, 15, 50, "Maternal Age");
    if (wMa) list.push(wMa);

    const wWt = checkRange(prePregnancyWeight, 35, 150, "Weight");
    if (wWt) list.push(wWt);
    
    // Note: Height check (50-300) implied by BMI and weight limits, but can add if needed.
    
    if (calculatedBmi) {
        if (calculatedBmi < 14.5 || calculatedBmi > 54) list.push(`BMI (${calculatedBmi}) is outside expected range (14.5-54)`);
    }

    const wGain = checkRange(weightGain, -28, 31, "Weight Gain");
    if (wGain) list.push(wGain);

    const wGa = checkRange(gestationalAge, 1, 42, "Gestational Age");
    if (wGa) list.push(wGa);

    return list;
  }, [maternalAge, prePregnancyWeight, weightGain, gestationalAge, calculatedBmi]);


  // Calculate BMI automatically
  useEffect(() => {
    const w = parseFloat(prePregnancyWeight);
    const h = parseFloat(height);
    if (!isNaN(w) && !isNaN(h) && h > 0) {
      // Height in cm to meters
      const h_m = h / 100;
      const bmi = w / (h_m * h_m);
      setCalculatedBmi(parseFloat(bmi.toFixed(1)));
    } else {
      setCalculatedBmi(null);
    }
  }, [prePregnancyWeight, height]);

  const isFormValid = useMemo(() => {
    const fields = [maternalAge, prePregnancyWeight, height, weightGain, gestationalAge];
    return fields.every(field => field.trim() !== "" && !isNaN(Number(field)));
  }, [maternalAge, prePregnancyWeight, height, weightGain, gestationalAge]);

  const handlePrediction = async (isAdvanced: boolean) => {
    setFormError("");
    if (!isFormValid) {
      setFormError("Please fill in all fields with valid numbers.");
      return;
    }

    // Block submission if warnings exist (strict mode as per backend)
    if (warnings.length > 0) {
        setFormError("Please correct the warnings above before predicting.");
        return;
    }

    setIsLoading(true);
    setError("");

    const baseFormData: PredictionFormData = { 
        maternalAge, 
        prePregnancyWeight, 
        height, 
        weightGain, 
        gestationalAge, // This is Weeks now
        conceptionArt,
        bmi: calculatedBmi || undefined
    };

    try {
      let result: PredictionHistoryItem;
      if (isAdvanced) {
        const advancedData: AdvancedPredictionFormData = { ...baseFormData, sdgOption, algorithmOption };
        result = await predictAdvanced(advancedData);
      } else {
        // The "Optimal Model" is now SVM SMOTE + XGBoost
        result = await predictOptimal(baseFormData);
      }
      addPrediction(result);
    } catch (err: unknown) {
      const modelName = isAdvanced ? `${sdgOption}+${algorithmOption}` : "Optimal";
      console.error(`Prediction failed for model ${modelName}:`, err);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const latestPrediction = history.length > 0 ? history[0] : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Predict GDM</h1>
        <p className="text-muted-foreground mt-2">
          Enter patient data for gestational diabetes prediction
        </p>
      </div>

      <Card className="bg-card border border-border">
        <CardContent className="pt-6">
          {/* Input Form Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="maternal-age">Maternal Age {checkRange(maternalAge, 15, 50, "") && <span className="text-yellow-500 text-xs ml-2">(15-50)</span>}</Label>
              <Input 
                id="maternal-age" 
                placeholder="e.g., 18-50" 
                className={clsx("h-12 bg-input border-border", checkRange(maternalAge, 15, 50, "") && "border-yellow-500 focus-visible:ring-yellow-500")}
                value={maternalAge} 
                onChange={(e) => setMaternalAge(e.target.value)} 
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="pre-pregnancy-weight">Pre-pregnancy Weight (kg) {checkRange(prePregnancyWeight, 35, 150, "") && <span className="text-yellow-500 text-xs ml-2">(35-150)</span>}</Label>
              <Input 
                id="pre-pregnancy-weight" 
                placeholder="e.g., 50-100" 
                className={clsx("h-12 bg-input border-border", checkRange(prePregnancyWeight, 35, 150, "") && "border-yellow-500 focus-visible:ring-yellow-500")}
                value={prePregnancyWeight} 
                onChange={(e) => setPrePregnancyWeight(e.target.value)} 
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="height">Height (cm)</Label>
              <Input id="height" placeholder="e.g., 140-180" className="h-12 bg-input border-border" value={height} onChange={(e) => setHeight(e.target.value)} />
              {calculatedBmi !== null && (
                <div className="flex items-center gap-2 mt-1">
                     <p className="text-sm text-muted-foreground">Calculated BMI: <span className="font-semibold text-foreground">{calculatedBmi}</span></p>
                     {(calculatedBmi < 14.5 || calculatedBmi > 54) && <span className="text-yellow-500 text-xs font-medium flex items-center"><AlertTriangle className="w-3 h-3 mr-1"/> Range: 14.5-54</span>}
                </div>
              )}
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="weight-gain">Weight Gain (kg) {checkRange(weightGain, -28, 31, "") && <span className="text-yellow-500 text-xs ml-2">(-28 to 31)</span>}</Label>
              <Input 
                id="weight-gain" 
                placeholder="e.g., 5-25" 
                className={clsx("h-12 bg-input border-border", checkRange(weightGain, -28, 31, "") && "border-yellow-500 focus-visible:ring-yellow-500")}
                value={weightGain} 
                onChange={(e) => setWeightGain(e.target.value)} 
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="gestational-age">Gestational Age (weeks) {checkRange(gestationalAge, 1, 42, "") && <span className="text-yellow-500 text-xs ml-2">(1-42)</span>}</Label>
              <Input 
                id="gestational-age" 
                placeholder="e.g., 1-42" 
                className={clsx("h-12 bg-input border-border", checkRange(gestationalAge, 1, 42, "") && "border-yellow-500 focus-visible:ring-yellow-500")}
                value={gestationalAge} 
                onChange={(e) => setGestationalAge(e.target.value)} 
              />
            </div>

            <div className="space-y-2 flex items-center pt-8">
               <Checkbox 
                 id="conception-art" 
                 checked={conceptionArt} 
                 onCheckedChange={(checked) => setConceptionArt(checked as boolean)}
                 className="mr-3 h-5 w-5"
               />
               <Label htmlFor="conception-art" className="cursor-pointer">Conception via ART (IVF/ICSI)?</Label>
            </div>
          </div>
          
          {/* Warning Messages Area */}
          <AnimatePresence>
            {warnings.length > 0 && (
                <motion.div 
                    initial={{ height: 0, opacity: 0 }} 
                    animate={{ height: "auto", opacity: 1 }} 
                    exit={{ height: 0, opacity: 0 }}
                    className="mt-6 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4"
                >
                    <div className="flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                        <div className="text-sm text-yellow-600 dark:text-yellow-400">
                            <p className="font-semibold mb-1">Please check the following fields:</p>
                            <ul className="list-disc pl-4 space-y-1">
                                {warnings.map((w, i) => <li key={i}>{w}</li>)}
                            </ul>
                        </div>
                    </div>
                </motion.div>
            )}
          </AnimatePresence>

        </CardContent>
      </Card>

      {/* --- Main Action Buttons --- */}
      <div className="space-y-4">
        <Button
          onClick={() => handlePrediction(false)}
          disabled={isLoading || !isFormValid || warnings.length > 0}
          className="w-full cursor-pointer bg-gradient-to-r from-ctgan-start to-ctgan-end text-white font-bold text-lg py-6 transition-transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-6 w-6 animate-spin" />
              {loadingSeconds > 5 ? (
                <span className="text-sm">System waking up... This initial analysis may take ~60 seconds. Please do not refresh.</span>
              ) : (
                "Predicting..."
              )}
            </>
          ) : (
            "Predict (Optimal Model)"
          )}
        </Button>

        <Button variant="outline" onClick={() => setIsAdvancedOpen(!isAdvancedOpen)} className="w-full cursor-pointer">
          Advanced Options
          <ChevronDown className={`ml-2 h-4 w-4 transition-transform ${isAdvancedOpen ? "rotate-180" : ""}`} />
        </Button>
      </div>

      {/* --- Advanced Options Section --- */}
      <AnimatePresence>
        {isAdvancedOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <Card className="bg-card/50 border-border">
              <CardContent className="pt-6 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* SDG Option Dropdown */}
                  <div className="space-y-2">
                    <Label>Synthetic Data Generation (SDG)</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between cursor-pointer">
                          <span className={clsx(sdgOption === OPTIMAL_SDG && "text-optimal font-semibold")}>{sdgOption}</span>
                          <ChevronDown className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-[--radix-dropdown-menu-trigger-width]">
                        <DropdownMenuRadioGroup value={sdgOption} onValueChange={setSdgOption}>
                          {SDG_OPTIONS.map(opt => (
                            <DropdownMenuRadioItem key={opt} value={opt} className={clsx("cursor-pointer", opt === OPTIMAL_SDG && "text-optimal font-semibold")}>
                              {opt}
                            </DropdownMenuRadioItem>
                          ))}
                        </DropdownMenuRadioGroup>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {/* Algorithm Option Dropdown */}
                  <div className="space-y-2">
                    <Label>ML Algorithm</Label>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" className="w-full justify-between cursor-pointer">
                          <span className={clsx(algorithmOption === OPTIMAL_ALGORITHM && "text-optimal font-semibold")}>{algorithmOption}</span>
                          <ChevronDown className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="w-[--radix-dropdown-menu-trigger-width]">
                        <DropdownMenuRadioGroup value={algorithmOption} onValueChange={setAlgorithmOption}>
                          {ALGORITHM_OPTIONS.map(opt => (
                            <DropdownMenuRadioItem key={opt} value={opt} className={clsx("cursor-pointer", opt === OPTIMAL_ALGORITHM && "text-optimal font-semibold")}>
                              {opt}
                            </DropdownMenuRadioItem>
                          ))}
                        </DropdownMenuRadioGroup>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
                <Button
                  onClick={() => handlePrediction(true)}
                  disabled={isLoading || !isFormValid || warnings.length > 0}
                  className="w-full cursor-pointer bg-gradient-to-r from-smote-start to-smote-end text-white font-bold text-lg py-6 transition-transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-6 w-6 animate-spin" />
                      {loadingSeconds > 5 ? (
                        <span className="text-sm">System waking up... This initial analysis may take ~60 seconds. Please do not refresh.</span>
                      ) : (
                        "Predicting..."
                      )}
                    </>
                  ) : (
                    "Predict with Selected Models"
                  )}
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {formError && <p className="text-yellow-500 text-center">{formError}</p>}
      {error && <p className="text-red-500 text-center">{error}</p>}

      {/* Prediction History */}
      {latestPrediction && !isLoading && (
        <div>
          <h2 className="text-2xl font-bold">Latest Prediction</h2>
          <div className="mt-4">
            <PredictionHistoryCard item={latestPrediction} />
          </div>
        </div>
      )}
      <div>
        <h2 className="text-2xl font-bold">Recent Predictions</h2>
        <div className="space-y-4 mt-4">
          {history.slice(1, 4).map((item) => (
            <PredictionHistoryCard key={item.id} item={item} />
          ))}
        </div>
      </div>
    </div>
  );
}