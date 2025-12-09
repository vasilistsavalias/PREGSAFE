"use client";

import React, { createContext, useContext, useState, ReactNode, useEffect } from "react";
import { getHistory, deleteHistoryItem, PredictionHistoryItem } from "@/lib/api";

interface PredictionContextType {
  history: PredictionHistoryItem[];
  addPrediction: (prediction: PredictionHistoryItem) => void;
  deletePrediction: (id: number | string) => void;
}

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const dbHistory = await getHistory();
        setHistory(dbHistory);
      } catch (error) {
        console.error("Failed to load initial history:", error);
      }
    };

    loadHistory();
  }, []);

  const addPrediction = (newPrediction: PredictionHistoryItem) => {
    setHistory((prevHistory) => [newPrediction, ...prevHistory]);
  };

  const deletePrediction = async (id: number | string) => {
    try {
      await deleteHistoryItem(id);
      setHistory((prevHistory) => prevHistory.filter((item) => item.id !== id));
    } catch (error) {
      console.error(`Failed to delete prediction with id: ${id}`, error);
    }
  };

  return (
    <PredictionContext.Provider value={{ history, addPrediction, deletePrediction }}>
      {children}
    </PredictionContext.Provider>
  );
}

export function usePredictionHistory() {
  const context = useContext(PredictionContext);
  if (context === undefined) {
    throw new Error("usePredictionHistory must be used within a PredictionProvider");
  }
  return context;
}

// Re-export the type for convenience
export type { PredictionHistoryItem } from "@/lib/api";