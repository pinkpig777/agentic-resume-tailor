import { SlidersHorizontal } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <SlidersHorizontal className="h-4 w-4" />
          Settings
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">Tune the pipeline.</h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Configure ingest behavior, generation thresholds, and tuning presets
          for each run.
        </p>
      </header>

      <Card className="animate-rise">
        <CardHeader>
          <CardTitle className="text-base">Coming next</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          Hook the settings form to the FastAPI endpoints and persist defaults
          across local runs.
        </CardContent>
      </Card>
    </div>
  );
}
