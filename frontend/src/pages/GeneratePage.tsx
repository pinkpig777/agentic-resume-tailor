import { useEffect, useMemo, useRef, useState } from "react";
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
  parentType: "experience" | "project" | "unknown";
  parentId: string;
  text: string;
  baseText: string;
  originalText: string;
  hasRewrite: boolean;
};

type StatusTone = "success" | "error";

type StatusMessage = {
  tone: StatusTone;
  message: string;
};

type RunProgressEvent = {
  stage?: string;
  status?: "pending" | "running" | "complete" | "error";
  iteration?: number;
  max_iters?: number;
  message?: string;
};

type BulletGroup = {
  key: string;
  title: string;
  section: "Experience" | "Project" | "Unknown";
  items: BulletCard[];
};

const LOOP_STAGES = [
  { key: "query", label: "Query" },
  { key: "retrieve", label: "Retrieve" },
  { key: "select", label: "Select" },
  { key: "rewrite", label: "Rewrite" },
  { key: "score", label: "Score" },
  { key: "render", label: "Render" },
];

const STAGE_INDEX = new Map(
  LOOP_STAGES.map((stage, idx) => [stage.key, idx]),
);
STAGE_INDEX.set("done", LOOP_STAGES.length - 1);

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
  edits: Record<string, string>,
) =>
  ids.map((id) => {
    const info = lookup.get(id);
    const rewrite = rewrites.get(id);
    const parentType = info?.parentType ?? "unknown";
    const section =
      parentType === "project"
        ? "Project"
        : parentType === "experience"
          ? "Experience"
          : "Unknown";
    const baseText = rewrite?.rewritten ?? info?.text ?? "";
    return {
      id,
      label: info?.label ?? "Unknown",
      section,
      parentType,
      parentId: info?.parentId ?? "",
      text: edits[id] ?? baseText,
      baseText,
      originalText: rewrite?.original ?? info?.text ?? "",
      hasRewrite: Boolean(rewrite && rewrite.rewritten !== rewrite.original),
    } satisfies BulletCard;
  });

const buildBulletGroups = (
  cards: BulletCard[],
  resumeData?: ResumeData,
): BulletGroup[] => {
  const cardsById = new Map(cards.map((card) => [card.id, card]));
  const idsByParent = new Map<string, string[]>();
  const unknownIds: string[] = [];

  cards.forEach((card) => {
    if (!card.parentId || card.parentType === "unknown") {
      unknownIds.push(card.id);
      return;
    }
    const key = `${card.parentType}:${card.parentId}`;
    const existing = idsByParent.get(key) ?? [];
    existing.push(card.id);
    idsByParent.set(key, existing);
  });

  const groups: BulletGroup[] = [];
  const pushGroup = (
    key: string,
    section: BulletGroup["section"],
    title: string,
  ) => {
    const ids = idsByParent.get(key);
    if (!ids?.length) {
      return;
    }
    groups.push({
      key,
      section,
      title,
      items: ids.map((id) => cardsById.get(id)).filter(Boolean) as BulletCard[],
    });
  };

  if (resumeData) {
    resumeData.experiences.forEach((exp) => {
      const title = `${exp.role} · ${exp.company}`.trim();
      pushGroup(`experience:${exp.job_id}`, "Experience", title || "Experience");
    });
    resumeData.projects.forEach((proj) => {
      pushGroup(`project:${proj.project_id}`, "Project", proj.name || "Project");
    });
  } else {
    const experienceKeys: string[] = [];
    const projectKeys: string[] = [];
    for (const key of idsByParent.keys()) {
      if (key.startsWith("experience:")) {
        experienceKeys.push(key);
      } else if (key.startsWith("project:")) {
        projectKeys.push(key);
      }
    }
    experienceKeys.forEach((key) => pushGroup(key, "Experience", "Experience"));
    projectKeys.forEach((key) => pushGroup(key, "Project", "Project"));
  }

  if (unknownIds.length) {
    groups.push({
      key: "unknown",
      section: "Unknown",
      title: "Other",
      items: unknownIds
        .map((id) => cardsById.get(id))
        .filter(Boolean) as BulletCard[],
    });
  }

  return groups;
};

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
  const [editedBullets, setEditedBullets] = useState<Record<string, string>>({});
  const [pdfNonce, setPdfNonce] = useState(() => Date.now());
  const [loopStage, setLoopStage] = useState<number | null>(null);
  const [loopStatus, setLoopStatus] = useState<RunProgressEvent["status"]>("pending");
  const [loopIteration, setLoopIteration] = useState<number | null>(null);
  const [loopMaxIters, setLoopMaxIters] = useState<number | null>(null);
  const [showPreview, setShowPreview] = useState(true);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const lastRunIdRef = useRef<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

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
  const pdfPreviewUrl = result ? `${pdfUrl}?v=${pdfNonce}` : "#";
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

  useEffect(() => {
    if (!report?.run_id || !report.selected_ids) {
      return;
    }
    if (lastRunIdRef.current === report.run_id) {
      return;
    }
    const hasLookup = report.selected_ids.some((id) => bulletLookup.has(id));
    const hasRewrites = report.selected_ids.some((id) => rewrites.has(id));
    if (!hasLookup && !hasRewrites) {
      return;
    }
    lastRunIdRef.current = report.run_id;
    const next: Record<string, string> = {};
    report.selected_ids.forEach((id) => {
      const info = bulletLookup.get(id);
      const rewrite = rewrites.get(id);
      next[id] = rewrite?.rewritten ?? info?.text ?? "";
    });
    setEditedBullets(next);
  }, [report?.run_id, report?.selected_ids, bulletLookup, rewrites]);

  useEffect(() => {
    if (result?.run_id) {
      setPdfNonce(Date.now());
    }
  }, [result?.run_id]);


  const selectedIds = useMemo(
    () => selectionOrder.filter((id) => selectedMap[id]),
    [selectionOrder, selectedMap],
  );

  const bulletCards = useMemo(
    () => buildBulletCards(selectionOrder, bulletLookup, rewrites, editedBullets),
    [selectionOrder, bulletLookup, rewrites, editedBullets],
  );

  const bulletGroups = useMemo(
    () => buildBulletGroups(bulletCards, resumeData),
    [bulletCards, resumeData],
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

  const editsDirty = useMemo(
    () =>
      bulletCards.some(
        (card) =>
          selectedMap[card.id] &&
          editedBullets[card.id] !== undefined &&
          editedBullets[card.id] !== card.baseText,
      ),
    [bulletCards, editedBullets, selectedMap],
  );

  const closeEventSource = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const handleProgressEvent = (payload: RunProgressEvent) => {
    if (payload.max_iters !== undefined) {
      setLoopMaxIters(payload.max_iters ?? null);
    }
    if (payload.iteration !== undefined) {
      setLoopIteration(
        payload.iteration !== null ? payload.iteration + 1 : null,
      );
    }
    if (payload.status) {
      setLoopStatus(payload.status);
    }
    if (payload.stage) {
      const idx = STAGE_INDEX.get(payload.stage);
      if (idx !== undefined) {
        setLoopStage(idx);
      }
    }
    if (payload.status === "complete") {
      setLoopStage(LOOP_STAGES.length - 1);
      closeEventSource();
    }
    if (payload.status === "error") {
      setStatus({
        tone: "error",
        message: payload.message || "Generation failed. Check the API.",
      });
      setLoopStage(null);
      closeEventSource();
    }
  };

  const startProgressStream = (runId: string) => {
    closeEventSource();
    const url = `${API_BASE_URL}/runs/${runId}/events`;
    const source = new EventSource(url);
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as RunProgressEvent;
        handleProgressEvent(payload);
      } catch {
        // Ignore malformed progress events.
      }
    };
    source.onerror = () => {
      setLoopStatus("error");
      setLoopStage(null);
      setStatus({
        tone: "error",
        message: "Progress stream interrupted. The run may still be processing.",
      });
      closeEventSource();
    };
    eventSourceRef.current = source;
  };

  useEffect(() => {
    return () => {
      closeEventSource();
    };
  }, []);

  const mutation = useMutation({
    mutationFn: (payload: { text: string; runId: string }) =>
      generateResume(payload.text, payload.runId),
    onMutate: ({ runId }) => {
      setResult(null);
      setSelectionOrder([]);
      setSelectedMap({});
      setEditedBullets({});
      setLoopStage(0);
      setLoopStatus("running");
      setLoopIteration(null);
      setLoopMaxIters(null);
      setStatus(null);
      startProgressStream(runId);
    },
    onSuccess: (data) => {
      setResult(data);
      setLoopStatus("complete");
      setLoopStage(LOOP_STAGES.length - 1);
      closeEventSource();
    },
    onError: () => {
      setStatus({ tone: "error", message: "Generation failed. Check the API." });
      setLoopStatus("error");
      setLoopStage(null);
      closeEventSource();
    },
  });

  const renderMutation = useMutation({
    mutationFn: (payload: {
      runId: string;
      selectedIds: string[];
      rewritten?: Record<string, string>;
      tempEdits?: Record<string, string>;
    }) =>
      renderSelection(payload.runId, {
        selected_ids: payload.selectedIds,
        rewritten_bullets: payload.rewritten,
        temp_overrides: payload.tempEdits ? { edits: payload.tempEdits } : undefined,
      }),
    onSuccess: () => {
      setStatus({ tone: "success", message: "PDF re-rendered." });
      setPdfNonce(Date.now());
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
    const runId =
      window.crypto?.randomUUID?.() ||
      `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    mutation.mutate({ text: trimmed, runId });
  };

  const toggleInclude = (id: string) => {
    setSelectedMap((current) => ({ ...current, [id]: !current[id] }));
  };

  const toggleOriginal = (id: string) => {
    setShowOriginal((current) => ({ ...current, [id]: !current[id] }));
  };

  const handleEditBullet = (id: string, value: string) => {
    setEditedBullets((current) => ({ ...current, [id]: value }));
  };

  const handleApplySelection = () => {
    if (!runId || !selectedIds.length) {
      return;
    }
    const rewritten = Object.fromEntries(
      bulletCards
        .filter((card) => card.hasRewrite && selectedMap[card.id])
        .map((card) => [card.id, card.baseText]),
    );
    const tempEdits = Object.fromEntries(
      bulletCards
        .filter((card) => selectedMap[card.id])
        .filter((card) => {
          const edited = editedBullets[card.id];
          return edited !== undefined && edited.trim() && edited !== card.baseText;
        })
        .map((card) => [card.id, editedBullets[card.id] as string]),
    );
    renderMutation.mutate({
      runId,
      selectedIds,
      rewritten: Object.keys(rewritten).length ? rewritten : undefined,
      tempEdits: Object.keys(tempEdits).length ? tempEdits : undefined,
    });
  };

  const showBackendWarning = settingsError || resumeError || reportError;
  const activeStage = loopStage ?? -1;
  const progressPercent =
    activeStage >= 0
      ? Math.round(((activeStage + 1) / LOOP_STAGES.length) * 100)
      : 0;
  const loopStatusLabel =
    loopStatus === "running"
      ? "Running"
      : loopStatus === "complete"
        ? "Complete"
        : loopStatus === "error"
          ? "Error"
          : "Pending";
  const iterationLabel =
    loopIteration && loopMaxIters
      ? `Iteration ${loopIteration}/${loopMaxIters}`
      : loopIteration
        ? `Iteration ${loopIteration}`
        : null;

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
          {loopStage !== null ? (
            <div className="rounded-lg border bg-muted/40 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                <span>Loop progress</span>
                <span>{loopStatusLabel}</span>
              </div>
              {iterationLabel ? (
                <div className="mt-2 text-xs font-medium text-muted-foreground">
                  {iterationLabel}
                </div>
              ) : null}
              <div className="mt-3 grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
                {LOOP_STAGES.map((stage, idx) => (
                  <div
                    key={stage.key}
                    className={cn(
                      "flex items-center gap-2 text-xs font-medium",
                      idx <= activeStage ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    <span
                      className={cn(
                        "h-2.5 w-2.5 rounded-full border",
                        idx < activeStage
                          ? "border-primary bg-primary"
                          : idx === activeStage
                            ? "border-accent bg-accent animate-pulse"
                            : "border-muted-foreground/40",
                      )}
                    />
                    <span>{stage.label}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>
          ) : null}
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
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Selected bullets</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
                <span>
                  Toggle bullets for this run and re-render the PDF when ready.
                </span>
                <span>{selectedIds.length} selected</span>
              </div>
              <div className="space-y-4">
                {bulletGroups.map((group) => (
                  <div
                    key={group.key}
                    className="rounded-xl border bg-card/60 p-4 shadow-sm"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="space-y-1">
                        <div className="text-xs uppercase tracking-[0.2em] text-muted-foreground">
                          {group.section}
                        </div>
                        <div className="text-sm font-semibold">{group.title}</div>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {group.items.filter((item) => selectedMap[item.id]).length}/
                        {group.items.length} selected
                      </div>
                    </div>
                    <div className="mt-4 space-y-3">
                      {group.items.map((card) => {
                        const isEdited = card.text !== card.baseText;
                        return (
                          <div
                            key={card.id}
                            className="rounded-lg border bg-background/80 p-3"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                {card.hasRewrite ? (
                                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[11px] text-emerald-700">
                                    Rewritten
                                  </span>
                                ) : null}
                                {isEdited ? (
                                  <span className="rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[11px] text-amber-700">
                                    Edited
                                  </span>
                                ) : null}
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
                              <Textarea
                                value={card.text}
                                onChange={(event) =>
                                  handleEditBullet(card.id, event.target.value)
                                }
                                className="min-h-[96px]"
                              />
                              {card.hasRewrite ? (
                                <button
                                  type="button"
                                  className="text-xs text-muted-foreground underline"
                                  onClick={() => toggleOriginal(card.id)}
                                >
                                  {showOriginal[card.id]
                                    ? "Hide original"
                                    : "Show original"}
                                </button>
                              ) : null}
                              {showOriginal[card.id] ? (
                                <div className="rounded-md border border-dashed bg-muted/40 p-2 text-xs text-muted-foreground">
                                  {card.originalText}
                                </div>
                              ) : null}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  variant="secondary"
                  onClick={handleApplySelection}
                  disabled={(!selectionDirty && !editsDirty) || renderMutation.isPending}
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
                {selectionDirty || editsDirty ? (
                  <span className="text-xs text-muted-foreground">
                    Updates pending. Re-render to refresh the PDF.
                  </span>
                ) : null}
              </div>
            </CardContent>
          </Card>

          <div className="rounded-xl border bg-card/60 shadow-sm">
            <button
              type="button"
              onClick={() => setShowPreview((current) => !current)}
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold"
              aria-expanded={showPreview}
            >
              <span className="uppercase tracking-[0.2em] text-muted-foreground">
                PDF preview
              </span>
              <span className="text-xs text-muted-foreground">
                {showPreview ? "Collapse" : "Expand"}
              </span>
            </button>
            {showPreview ? (
              <div className="border-t px-4 pb-4 pt-3">
                <div className="space-y-3">
                  <iframe
                    title="Resume preview"
                    src={pdfPreviewUrl}
                    className="h-[640px] w-full rounded-md border"
                  />
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
                </div>
              </div>
            ) : null}
          </div>

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

          <details className="rounded-lg border bg-card p-4 text-sm">
            <summary className="cursor-pointer text-sm font-medium">
              Technical details
            </summary>
            <div className="mt-3 space-y-2 text-xs text-muted-foreground">
              <div>Run ID: {result.run_id}</div>
              <div>Best iteration: {report?.best_iteration_index ?? "n/a"}</div>
              <div>Score: {report?.best_score?.final_score ?? "n/a"}</div>
              <div>Report URL: {result.report_url}</div>
            </div>
          </details>
        </div>
      ) : null}
    </div>
  );
}
