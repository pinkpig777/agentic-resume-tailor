import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Download, Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { API_BASE_URL, generateResume } from "@/lib/api";
import type { GenerateResponse } from "@/types/schema";

export default function GeneratePage() {
  const [jdText, setJdText] = useState("");
  const [result, setResult] = useState<GenerateResponse | null>(null);

  const mutation = useMutation({
    mutationFn: generateResume,
    onMutate: () => setResult(null),
    onSuccess: (data) => setResult(data),
  });

  const handleGenerate = () => {
    const trimmed = jdText.trim();
    if (!trimmed || mutation.isPending) {
      return;
    }
    mutation.mutate(trimmed);
  };

  const pdfUrl = result ? new URL(result.pdf_url, API_BASE_URL).toString() : "";

  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <Sparkles className="h-4 w-4" />
          Generate
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">
          Tailor a resume in minutes.
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Paste a job description, run the agent loop, and grab the PDF when
          it is ready.
        </p>
      </header>

      <Card className="animate-rise">
        <CardHeader>
          <CardTitle>Job description</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            value={jdText}
            onChange={(event) => setJdText(event.target.value)}
            placeholder="Paste the job description here..."
            className="min-h-[220px]"
          />
          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={handleGenerate}
              disabled={mutation.isPending || !jdText.trim()}
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating
                </>
              ) : (
                "Generate"
              )}
            </Button>
            {mutation.isError ? (
              <span className="text-sm text-destructive">
                Generation failed. Check the API server.
              </span>
            ) : null}
          </div>
        </CardContent>
      </Card>

      {result ? (
        <Card className="animate-rise animate-rise-delay-1">
          <CardHeader>
            <CardTitle>Run ready</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm text-muted-foreground">Run ID</div>
            <div className="font-mono text-sm">{result.run_id}</div>
            <div className="flex flex-wrap items-center gap-3">
              <Button asChild>
                <a href={pdfUrl} target="_blank" rel="noreferrer">
                  <Download className="h-4 w-4" />
                  Download PDF
                </a>
              </Button>
              <span className="text-xs text-muted-foreground">
                Profile used: {result.profile_used ? "yes" : "no"}
              </span>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
