import DashboardLayout from "@/components/DashboardLayout";
import { PredictionProvider } from "@/context/PredictionContext";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <PredictionProvider>
      <DashboardLayout>{children}</DashboardLayout>
    </PredictionProvider>
  );
}
