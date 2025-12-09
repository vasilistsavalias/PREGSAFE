"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Trash2, Activity, Clock, Hash } from "lucide-react";
import clsx from "clsx";
import { usePredictionHistory } from "@/context/PredictionContext";
import type { PredictionHistoryItem } from "@/lib/api";

interface PredictionHistoryCardProps {
  item: PredictionHistoryItem;
}

function toAthensHHMM_AMPM(utcTimestamp: string | undefined | null): string {
  if (!utcTimestamp) return "";

  let s = utcTimestamp.trim();
  const hasTZ = /[zZ]$/.test(s) || /[+\-]\d{2}:\d{2}$/.test(s) || /[+\-]\d{4}$/.test(s);

  if (!hasTZ) {
    s = s.replace(" ", "T") + "Z";
  }

  const d = new Date(s);
  if (isNaN(d.getTime())) return utcTimestamp;

  return d.toLocaleTimeString("en-US", {
    timeZone: "Europe/Athens",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

export default function PredictionHistoryCard({ item }: PredictionHistoryCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { deletePrediction } = usePredictionHistory();

  if (!item || !item.id) {
    return null;
  }

  const isPositive = item.prediction_result === "Positive";
  const isUncertain = item.prediction_result === "Uncertain";
  const randomId = item.id.toString().substring(0, 4).toUpperCase();

  // Color logic
  const statusColors = {
    border: isPositive ? "border-red-500/30" : isUncertain ? "border-yellow-500/30" : "border-green-500/30",
    iconBg: isPositive ? "bg-red-500/10 text-red-500" : isUncertain ? "bg-yellow-500/10 text-yellow-500" : "bg-green-500/10 text-green-500",
    text: isPositive ? "text-red-500" : isUncertain ? "text-yellow-500" : "text-green-500",
    score: isPositive ? "text-red-600 dark:text-red-400" : isUncertain ? "text-yellow-600 dark:text-yellow-400" : "text-green-600 dark:text-green-400"
  };

  return (
    <Card className={clsx("overflow-hidden transition-all", statusColors.border)}>
      {/* 
        Responsive Layout Strategy:
        Mobile: Stacked. Row 1 (Icon/Result + Score/Actions). Row 2 (Meta: ID/Time).
        Desktop: Single Row. Icon/Result -> ID/Time -> Score -> Actions.
      */}
      <div className="p-4">
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          
          {/* Main Content Area */}
          <div className="flex items-center justify-between w-full md:w-auto md:justify-start gap-4">
            
            {/* Icon & Result Group */}
            <div className="flex items-center gap-3 md:gap-4">
              <div className={clsx("w-10 h-10 md:w-12 md:h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors", statusColors.iconBg)}>
                <Activity className="w-5 h-5 md:w-6 md:h-6" />
              </div>
              
              <div className="flex flex-col">
                <span className={clsx("text-lg md:text-xl font-bold leading-tight", statusColors.text)}>
                  {item.prediction_result.replace("GDM ", "")}
                </span>
                <Badge variant="secondary" className="w-fit text-[10px] md:text-xs mt-0.5 px-1.5 py-0 h-5 font-normal text-muted-foreground border-border/50">
                  {item.model_used}
                </Badge>
              </div>
            </div>

            {/* Mobile: Score & Actions (Visible only on small screens) */}
            <div className="flex md:hidden items-center gap-3">
               <span className={clsx("text-lg font-bold font-mono", statusColors.score)}>
                  {item.confidence.toFixed(1)}%
               </span>
               <div className="flex gap-1">
                 <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-destructive" onClick={() => deletePrediction(item.id)}>
                    <Trash2 className="w-4 h-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={() => setIsExpanded(!isExpanded)}>
                    <motion.div animate={{ rotate: isExpanded ? 180 : 0 }}>
                      <ChevronDown className="w-4 h-4" />
                    </motion.div>
                  </Button>
               </div>
            </div>
          </div>

          {/* Desktop Meta & Actions (Hidden on Mobile) */}
          <div className="hidden md:flex flex-1 items-center justify-end gap-6">
             {/* ID */}
             <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Hash className="w-3.5 h-3.5 opacity-50" />
                <span className="font-mono">{randomId}</span>
             </div>
             
             {/* Time */}
             <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="w-3.5 h-3.5 opacity-50" />
                <span>{toAthensHHMM_AMPM(item.timestamp)}</span>
             </div>

             {/* Score */}
             <div className="text-right">
                <span className="text-xs text-muted-foreground block leading-none mb-1">Risk Score</span>
                <span className={clsx("text-xl font-bold font-mono", statusColors.score)}>
                   {item.confidence.toFixed(1)}%
                </span>
             </div>

             {/* Actions */}
             <div className="flex items-center gap-1 pl-2 border-l border-border/50">
                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-destructive" onClick={() => deletePrediction(item.id)}>
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="sm" className="h-8 px-2 text-muted-foreground" onClick={() => setIsExpanded(!isExpanded)}>
                  {isExpanded ? "Hide" : "Details"}
                  <motion.div animate={{ rotate: isExpanded ? 180 : 0 }} className="ml-1">
                    <ChevronDown className="w-4 h-4" />
                  </motion.div>
                </Button>
             </div>
          </div>

          {/* Mobile Meta Row (ID & Time) - Visible only on mobile */}
          <div className="flex md:hidden items-center justify-between pt-3 mt-1 border-t border-border/40 text-xs text-muted-foreground">
             <div className="flex items-center gap-1.5">
                <Hash className="w-3 h-3 opacity-50" />
                <span className="font-mono">{randomId}</span>
             </div>
             <div className="flex items-center gap-1.5">
                <Clock className="w-3 h-3 opacity-50" />
                <span>{toAthensHHMM_AMPM(item.timestamp)}</span>
             </div>
          </div>

        </div>
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }} 
            animate={{ height: "auto", opacity: 1 }} 
            exit={{ height: 0, opacity: 0 }} 
            transition={{ duration: 0.2 }}
          >
            <CardContent className="p-4 pt-0 bg-muted/30 border-t border-border/40">
              <div className="grid grid-cols-2 gap-3 pt-4 text-sm">
                <div className="p-3 rounded-lg bg-background border border-border/50">
                  <span className="text-xs text-muted-foreground block mb-0.5">Maternal Age</span>
                  <span className="font-semibold">{item.maternal_age}</span>
                </div>
                <div className="p-3 rounded-lg bg-background border border-border/50">
                  <span className="text-xs text-muted-foreground block mb-0.5">Gestational Age</span>
                  <span className="font-semibold">{item.gestational_age_weeks} weeks</span>
                </div>
                <div className="p-3 rounded-lg bg-background border border-border/50">
                  <span className="text-xs text-muted-foreground block mb-0.5">Pre-preg Weight</span>
                  <span className="font-semibold">{item.pre_pregnancy_weight} kg</span>
                </div>
                <div className="p-3 rounded-lg bg-background border border-border/50">
                  <span className="text-xs text-muted-foreground block mb-0.5">Weight Gain</span>
                  <span className="font-semibold">{item.weight_gain} kg</span>
                </div>
                <div className="col-span-2 p-3 rounded-lg bg-background border border-border/50 flex items-center justify-between">
                  <div>
                    <span className="text-xs text-muted-foreground block mb-0.5">Height</span>
                    <span className="font-semibold">{item.height} cm</span>
                  </div>
                   {/* We could calculate BMI here too if we wanted to show it in history */}
                </div>
              </div>
            </CardContent>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}
