import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Loader2, RefreshCcw, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  API_BASE_URL,
  fetchData,
  fetchRunReport,
  fetchSettings,
  generateResume,
  renderSelection,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import type { GenerateResponse, ResumeData, RunReport } from "@/types/schema";

type BulletInfo = {
  text: string;
  label: string;
  parentType: "experience" | "project";
  parentId: string;
};

type BulletCard = {
  id: string;
  label: string;
  section: string;
  text: string;
  originalText: string;
  hasRewrite: boolean;
};

type StatusTone = "success" | "error";

type StatusMessage = {
  tone: StatusTone;
  message: string;
};

const buildBulletLookup = (data: ResumeData) => {
  const map = new Map<string, BulletInfo>();

  data.experiences.forEach((exp) => {
    const label = `${exp.role} · ${exp.company}`;
    exp.bullets.forEach((bullet) => {
      const id = `exp:${exp.job_id}:${bullet.id}`;
      map.set(id, {
        text: bullet.text_latex,
        label,
        parentType: "experience",
        parentId: exp.job_id,
      });
    });
  });

  data.projects.forEach((proj) => {
    const label = proj.name;
    proj.bullets.forEach((bullet) => {
      const id = `proj:${proj.project_id}:${bullet.id}`;
      map.set(id, {
        text: bullet.text_latex,
        label,
        parentType: "project",
        parentId: proj.project_id,
      });
    });
  });

  return map;
};

const buildRewriteMap = (report?: RunReport) => {
  const map = new Map<string, { rewritten: string; original: string }>();
  (report?.rewritten_bullets ?? []).forEach((entry) => {
    map.set(entry.bullet_id, {
      rewritten: entry.rewritten_text,
      original: entry.original_text,
    });
  });
  return map;
};

const buildBulletCards = (
  ids: string[],
  lookup: Map<string, BulletInfo>,
  rewrites: Map<string, { rewritten: string; original: string }>,
) =>
  ids.map((id) => {
    const info = lookup.get(id);
    const rewrite = rewrites.get(id);
    const section = info?.parentType === "project" ? "Project" : "Experience";
    return {
      id,
      label: info?.label ?? "Unknown",
      section,
      text: rewrite?.rewritten ?? info?.text ?? "",
      originalText: rewrite?.original ?? info?.text ?? "",
      hasRewrite: Boolean(rewrite && rewrite.rewritten !== rewrite.original),
    } satisfies BulletCard;
  });

const buildMissingSkills = (report?: RunReport) => ({
  must: report?.best_score?.must_missing_bullets_only ?? [],
  nice: report?.best_score?.nice_missing_bullets_only ?? [],
});

export default function GeneratePage() {
  const queryClient = useQueryClient();
  const [jdText, setJdText] = useState("");
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [selectionOrder, setSelectionOrder] = useState<string[]>([]);
  const [selectedMap, setSelectedMap] = useState<Record<string, boolean>>({});
  const [showOriginal, setShowOriginal] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<StatusMessage | null>(null);

  const { isError: settingsError, refetch: refetchSettings } = useQuery({
    queryKey: ["settings"],
    queryFn: fetchSettings,
  });

  const {
    data: resumeData,
    isError: resumeError,
    refetch: refetchResume,
  } = useQuery({
    queryKey: ["resumeData"],
    queryFn: fetchData,
  });

  const runId = result?.run_id;
  const reportUrl = result ? `${API_BASE_URL}${result.report_url}` : "#";
  const pdfUrl = result ? `${API_BASE_URL}${result.pdf_url}` : "#";
  const texUrl = result ? `${API_BASE_URL}${result.tex_url}` : "#";

  const {
    data: report,
    isError: reportError,
    refetch: refetchReport,
  } = useQuery({
    queryKey: ["runReport", runId],
    queryFn: () => fetchRunReport(runId as string),
    enabled: Boolean(runId),
  });

  useEffect(() => {
    if (!report?.selected_ids) {
      return;
    }
    setSelectionOrder(report.selected_ids);
    setSelectedMap(
      report.selected_ids.reduce(
        (acc, id) => ({ ...acc, [id]: true }),
        {} as Record<string, boolean>,
      ),
    );
    setShowOriginal({});
  }, [report?.run_id, report?.selected_ids]);

  const bulletLookup = useMemo(() => {
    if (!resumeData) {
      return new Map();
    }
    return buildBulletLookup(resumeData);
  }, [resumeData]);

  const rewrites = useMemo(() => buildRewriteMap(report), [report]);

  const selectedIds = useMemo(
    () => selectionOrder.filter((id) => selectedMap[id]),
    [selectionOrder, selectedMap],
  );

  const bulletCards = useMemo(
    () => buildBulletCards(selectionOrder, bulletLookup, rewrites),
    [selectionOrder, bulletLookup, rewrites],
  );

  const missingSkills = useMemo(() => buildMissingSkills(report), [report]);

  const selectionDirty = useMemo(() => {
    if (!report?.selected_ids) {
      return false;
    }
    if (report.selected_ids.length !== selectedIds.length) {
      return true;
    }
    return report.selected_ids.some((id, idx) => id !== selectedIds[idx]);
  }, [report?.selected_ids, selectedIds]);

  const mutation = useMutation({
    mutationFn: (text: string) => generateResume(text),
    onMutate: () => {
      setResult(null);
      setSelectionOrder([]);
      setSelectedMap({});
      setStatus(null);
    },
    onSuccess: (data) => setResult(data),
    onError: () => {
      setStatus({ tone: "error", message: "Generation failed. Check the API." });
    },
  });

  const renderMutation = useMutation({
    mutationFn: (payload: {
      runId: string;
      selectedIds: string[];
      rewritten?: Record<string, string>;
    }) =>
      renderSelection(payload.runId, {
        selected_ids: payload.selectedIds,
        rewritten_bullets: payload.rewritten,
      }),
    onSuccess: () => {
      setStatus({ tone: "success", message: "PDF re-rendered." });
      queryClient.invalidateQueries({ queryKey: ["runReport", runId] });
    },
    onError: () => {
      setStatus({ tone: "error", message: "Failed to render PDF." });
    },
  });

  const handleGenerate = () => {
    const trimmed = jdText.trim();
    if (!trimmed || mutation.isPending) {
      return;
    }
    mutation.mutate(trimmed);
  };

  const toggleInclude = (id: string) => {
    setSelectedMap((current) => ({ ...current, [id]: !current[id] }));
  };

  const toggleOriginal = (id: string) => {
    setShowOriginal((current) => ({ ...current, [id]: !current[id] }));
  };

  const handleApplySelection = () => {
    if (!runId || !selectedIds.length) {
      return;
    }
    const rewritten = Object.fromEntries(
      bulletCards
        .filter((card) => card.hasRewrite && selectedMap[card.id])
        .map((card) => [card.id, card.text]),
    );
    renderMutation.mutate({
      runId,
      selectedIds,
      rewritten: Object.keys(rewritten).length ? rewritten : undefined,
    });
  };

  const showBackendWarning = settingsError || resumeError || reportError;

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
          Paste a job description, run the agent loop, then review the suggested
          bullets before rendering a one-page PDF.
        </p>
      </header>

      {showBackendWarning ? (
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <span>We could not reach the API. Check the server and retry.</span>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              refetchSettings();
              refetchResume();
              if (runId) {
                refetchReport();
              }
            }}
          >
            <RefreshCcw className="h-4 w-4" />
            Retry
          </Button>
        </div>
      ) : null}

      <Card className="animate-rise">
        <CardHeader>
          <CardTitle>Job description</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Label htmlFor="jd-text">Paste a JD</Label>
          <Textarea
            id="jd-text"
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
              <Button variant="secondary" asChild>
                <a href={reportUrl} target="_blank" rel="noreferrer">
                  <Download className="h-4 w-4" />
                  Download report
                </a>
              </Button>
              <span className="text-xs text-muted-foreground">
                JD parser used: {result.profile_used ? "yes" : "no"}
              </span>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {status ? (
        <div
          className={cn(
            "rounded-lg border px-3 py-2 text-sm",
            status.tone === "success"
              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
              : "border-destructive/30 bg-destructive/10 text-destructive",
          )}
        >
          {status.message}
        </div>
      ) : null}

      {result ? (
        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Selected bullets</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                  <span>
                    Toggle bullets for this run and re-render the PDF when
                    ready.
                  </span>
                  <span>{selectedIds.length} selected</span>
                </div>
                <div className="space-y-4">
                  {bulletCards.map((card) => (
                    <div
                      key={card.id}
                      className="rounded-lg border bg-card p-4 shadow-sm"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="space-y-1">
                          <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                            {card.section}
                          </div>
                          <div className="text-sm font-medium">{card.label}</div>
                        </div>
                        <label className="flex items-center gap-2 text-xs font-medium">
                          <input
                            type="checkbox"
                            checked={Boolean(selectedMap[card.id])}
                            onChange={() => toggleInclude(card.id)}
                          />
                          Include
                        </label>
                      </div>
                      <div className="mt-3 space-y-2">
                        {card.hasRewrite ? (
                          <button
                            type="button"
                            className="text-xs text-muted-foreground underline"
                            onClick={() => toggleOriginal(card.id)}
                          >
                            {showOriginal[card.id]
                              ? "Show rewritten"
                              : "Show original"}
                          </button>
                        ) : null}
                        <p className="text-sm leading-relaxed">
                          {showOriginal[card.id]
                            ? card.originalText
                            : card.text}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    variant="secondary"
                    onClick={handleApplySelection}
                    disabled={!selectionDirty || renderMutation.isPending}
                  >
                    {renderMutation.isPending ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Rendering
                      </>
                    ) : (
                      "Re-render PDF"
                    )}
                  </Button>
                  {selectionDirty ? (
                    <span className="text-xs text-muted-foreground">
                      Selection changed. Re-render to update PDF.
                    </span>
                  ) : null}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Missing skills</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                    Must-have
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {missingSkills.must.length ? (
                      missingSkills.must.map((skill) => (
                        <span
                          key={skill}
                          className="rounded-full border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-700"
                        >
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-muted-foreground">
                        No missing must-haves detected.
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                    Nice-to-have
                  </div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {missingSkills.nice.length ? (
                      missingSkills.nice.map((skill) => (
                        <span
                          key={skill}
                          className="rounded-full border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600"
                        >
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-muted-foreground">
                        No missing nice-to-haves detected.
                      </span>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>PDF preview</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {result ? (
                  <iframe
                    title="Resume preview"
                    src={pdfUrl}
                    className="h-[560px] w-full rounded-md border"
                  />
                ) : (
                  <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
                    Generate a run to preview the PDF.
                  </div>
                )}
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" asChild>
                    <a href={pdfUrl} target="_blank" rel="noreferrer">
                      Download PDF
                    </a>
                  </Button>
                  <Button variant="secondary" asChild>
                    <a href={texUrl} target="_blank" rel="noreferrer">
                      Download TeX
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>

            <details className="rounded-lg border bg-card p-4 text-sm">
              <summary className="cursor-pointer text-sm font-medium">
                Technical details
              </summary>
              <div className="mt-3 space-y-2 text-xs text-muted-foreground">
                <div>Run ID: {result.run_id}</div>
                <div>Best iteration: {report?.best_iteration_index ?? "n/a"}</div>
                <div>
                  Score: {report?.best_score?.final_score ?? "n/a"}
                </div>
                <div>Report URL: {result.report_url}</div>
              </div>
            </details>
          </div>
        </div>
      ) : null}
    </div>
  );
}
