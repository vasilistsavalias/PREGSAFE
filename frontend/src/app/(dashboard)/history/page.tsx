"use client";

import { usePredictionHistory } from "@/context/PredictionContext";
import PredictionHistoryCard from "@/components/PredictionHistoryCard";
import { Card, CardContent } from "@/components/ui/card";

export default function HistoryPage() {
  const { history } = usePredictionHistory();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Prediction History</h1>
        <p className="text-muted-foreground mt-2">
          View all past predictions.
        </p>
      </div>

      {history.length === 0 ? (
        <Card className="bg-card border border-border flex items-center justify-center h-48">
          <CardContent className="pt-6">
            <p className="text-muted-foreground">No predictions yet</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {history.map((item) => (
            <PredictionHistoryCard key={item.id} item={item} />
          ))}
        </div>
      )}
    </div>
  );
}