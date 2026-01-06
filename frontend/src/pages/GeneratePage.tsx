import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  ChevronDown,
  Download,
  GripVertical,
  Loader2,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  API_BASE_URL,
  fetchData,
  fetchRunReport,
  generateResume,
  renderSelection,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  GenerateResponse,
  ResumeData,
  RunReport,
  TempAddition,
  TempOverrides,
} from "@/types/schema";

type SelectionItem = {
  id: string;
  text: string;
  originalText: string;
  label: string;
  parentType?: "experience" | "project";
  parentId?: string;
  tempId?: string;
  isTemp: boolean;
};

type StatusTone = "success" | "error";

type StatusMessage = {
  tone: StatusTone;
  message: string;
};

type ParentOption = {
  id: string;
  label: string;
};

type BulletOption = {
  id: string;
  text: string;
  parentType: "experience" | "project";
  parentId: string;
  parentLabel: string;
};

type BulletGroup = {
  key: string;
  label: string;
  parentType: "experience" | "project";
  parentId: string;
  items: BulletOption[];
};

type StoredGenerateState = {
  jdText: string;
  result: GenerateResponse | null;
  selection: SelectionItem[];
};

const STORAGE_KEY = "art.generate.state.v1";

const loadStoredState = (): StoredGenerateState | null => {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as StoredGenerateState;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return {
      jdText: parsed.jdText ?? "",
      result: parsed.result ?? null,
      selection: Array.isArray(parsed.selection) ? parsed.selection : [],
    };
  } catch {
    return null;
  }
};

const persistState = (state: StoredGenerateState) => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage write failures (private mode, quota, etc.)
  }
};

const buildBulletId = (
  parentType: "experience" | "project",
  parentId: string,
  localId: string,
) => `${parentType === "experience" ? "exp" : "proj"}:${parentId}:${localId}`;

const buildBulletLookup = (data: ResumeData) => {
  const map = new Map<
    string,
    { text: string; label: string; parentType: "experience" | "project"; parentId: string }
  >();

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

const buildAvailableGroups = (
  data: ResumeData,
  selectedIds: Set<string>,
): BulletGroup[] => {
  const groups: BulletGroup[] = [];

  data.experiences.forEach((exp) => {
    const label = `${exp.role} · ${exp.company}`;
    const items = exp.bullets
      .map((bullet) => ({
        id: `exp:${exp.job_id}:${bullet.id}`,
        text: bullet.text_latex,
        parentType: "experience" as const,
        parentId: exp.job_id,
        parentLabel: label,
      }))
      .filter((item) => !selectedIds.has(item.id));
    if (items.length) {
      groups.push({
        key: `exp:${exp.job_id}`,
        label,
        parentType: "experience",
        parentId: exp.job_id,
        items,
      });
    }
  });

  data.projects.forEach((proj) => {
    const label = proj.name;
    const items = proj.bullets
      .map((bullet) => ({
        id: `proj:${proj.project_id}:${bullet.id}`,
        text: bullet.text_latex,
        parentType: "project" as const,
        parentId: proj.project_id,
        parentLabel: label,
      }))
      .filter((item) => !selectedIds.has(item.id));
    if (items.length) {
      groups.push({
        key: `proj:${proj.project_id}`,
        label,
        parentType: "project",
        parentId: proj.project_id,
        items,
      });
    }
  });

  return groups;
};

const mapTempAdditions = (report: RunReport) => {
  const additions = report.temp_additions ?? [];
  const map = new Map<string, TempAddition>();
  additions.forEach((addition, index) => {
    const bulletId =
      addition.bullet_id ||
      buildBulletId(
        addition.parent_type,
        addition.parent_id,
        addition.temp_id || `tmp_${report.run_id}_${index + 1}`,
      );
    map.set(bulletId, { ...addition, bullet_id: bulletId });
  });
  return map;
};

const buildSelectionFromReport = (
  report: RunReport,
  lookup: Map<
    string,
    { text: string; label: string; parentType: "experience" | "project"; parentId: string }
  >,
) => {
  const additionsById = mapTempAdditions(report);
  const edits = report.temp_edits ?? {};

  return report.selected_ids.map((id) => {
    const addition = additionsById.get(id);
    if (addition) {
      const label =
        lookup.get(id)?.label ||
        `${addition.parent_type} · ${addition.parent_id}`;
      return {
        id,
        text: addition.text_latex,
        originalText: addition.text_latex,
        label,
        parentType: addition.parent_type,
        parentId: addition.parent_id,
        tempId: addition.temp_id,
        isTemp: true,
      } satisfies SelectionItem;
    }

    const info = lookup.get(id);
    const baseText = info?.text ?? "";
    const text = edits[id] ?? baseText;
    return {
      id,
      text,
      originalText: baseText,
      label: info?.label ?? "Unknown bullet",
      parentType: info?.parentType,
      parentId: info?.parentId,
      isTemp: false,
    } satisfies SelectionItem;
  });
};

function SortableSelectionRow({
  item,
  onChange,
  onBlur,
  onDelete,
}: {
  item: SelectionItem;
  onChange: (id: string, value: string) => void;
  onBlur: (id: string, value: string) => void;
  onDelete: (id: string) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: item.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "flex gap-3 rounded-lg border bg-card p-3 shadow-sm",
        isDragging && "opacity-70",
      )}
    >
      <button
        type="button"
        className="mt-1 inline-flex h-9 w-9 items-center justify-center rounded-md border bg-muted text-muted-foreground transition hover:text-foreground"
        aria-label="Drag bullet"
        {...attributes}
        {...listeners}
      >
        <GripVertical className="h-4 w-4" />
      </button>
      <div className="flex-1 space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span className="rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.2em]">
            {item.parentType === "project" ? "Project" : "Experience"}
          </span>
          <span className="font-medium text-foreground">{item.label}</span>
          {item.isTemp ? (
            <span className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] uppercase tracking-[0.2em] text-accent">
              Temp
            </span>
          ) : null}
        </div>
        <Textarea
          value={item.text}
          onChange={(event) => onChange(item.id, event.target.value)}
          onBlur={(event) => onBlur(item.id, event.target.value)}
          className="min-h-[84px]"
          placeholder="Bullet text"
        />
      </div>
      <button
        type="button"
        className="mt-1 inline-flex h-9 w-9 items-center justify-center rounded-md border bg-muted text-muted-foreground transition hover:bg-destructive/10 hover:text-destructive"
        aria-label="Remove bullet"
        onClick={() => onDelete(item.id)}
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </div>
  );
}

export default function GeneratePage() {
  const initialState = useMemo(() => loadStoredState(), []);
  const [jdText, setJdText] = useState(() => initialState?.jdText ?? "");
  const [result, setResult] = useState<GenerateResponse | null>(
    () => initialState?.result ?? null,
  );
  const [selection, setSelection] = useState<SelectionItem[]>(
    () => initialState?.selection ?? [],
  );
  const [seededRunId, setSeededRunId] = useState<string | null>(() => {
    if (initialState?.result?.run_id && initialState.selection?.length) {
      return initialState.result.run_id;
    }
    return null;
  });
  const [selectionStatus, setSelectionStatus] = useState<StatusMessage | null>(null);
  const [newBulletType, setNewBulletType] = useState<"experience" | "project">(
    "experience",
  );
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>(
    {},
  );
  const [newBulletParentId, setNewBulletParentId] = useState("");
  const [newBulletText, setNewBulletText] = useState("");
  const tempCounter = useRef(0);

  const { data: resumeData } = useQuery({
    queryKey: ["resumeData"],
    queryFn: fetchData,
  });

  const runId = result?.run_id;

  const {
    data: report,
    isLoading: reportLoading,
    isError: reportError,
  } = useQuery({
    queryKey: ["runReport", runId],
    queryFn: () => fetchRunReport(runId as string),
    enabled: Boolean(runId),
  });

  const bulletLookup = useMemo(() => {
    if (!resumeData) {
      return new Map();
    }
    return buildBulletLookup(resumeData);
  }, [resumeData]);

  const selectedIdSet = useMemo(
    () => new Set(selection.map((item) => item.id)),
    [selection],
  );

  const availableGroups = useMemo(() => {
    if (!resumeData) {
      return [];
    }
    return buildAvailableGroups(resumeData, selectedIdSet);
  }, [resumeData, selectedIdSet]);

  const experienceOptions = useMemo<ParentOption[]>(() => {
    if (!resumeData) {
      return [];
    }
    return resumeData.experiences.map((exp) => ({
      id: exp.job_id,
      label: `${exp.role} · ${exp.company}`,
    }));
  }, [resumeData]);

  const projectOptions = useMemo<ParentOption[]>(() => {
    if (!resumeData) {
      return [];
    }
    return resumeData.projects.map((proj) => ({
      id: proj.project_id,
      label: proj.name,
    }));
  }, [resumeData]);

  const parentOptions =
    newBulletType === "experience" ? experienceOptions : projectOptions;

  useEffect(() => {
    persistState({ jdText, result, selection });
  }, [jdText, result, selection]);

  useEffect(() => {
    if (!parentOptions.length) {
      setNewBulletParentId("");
      return;
    }
    if (!parentOptions.some((option) => option.id === newBulletParentId)) {
      setNewBulletParentId(parentOptions[0].id);
    }
  }, [parentOptions, newBulletParentId]);

  useEffect(() => {
    if (!report || !resumeData) {
      return;
    }
    if (report.run_id === seededRunId) {
      return;
    }
    setSelection(buildSelectionFromReport(report, bulletLookup));
    setSeededRunId(report.run_id);
  }, [report, resumeData, bulletLookup, seededRunId]);

  const mutation = useMutation({
    mutationFn: generateResume,
    onMutate: () => {
      setResult(null);
      setSelection([]);
      setSeededRunId(null);
      setSelectionStatus(null);
    },
    onSuccess: (data) => setResult(data),
  });

  const renderMutation = useMutation({
    mutationFn: ({
      runId,
      payload,
    }: {
      runId: string;
      payload: { selected_ids: string[]; temp_overrides?: TempOverrides };
    }) => renderSelection(runId, payload),
    onSuccess: () => {
      setSelectionStatus({
        tone: "success",
        message: "PDF re-rendered with your selection.",
      });
    },
    onError: () => {
      setSelectionStatus({
        tone: "error",
        message: "Failed to render PDF. Check the API server.",
      });
    },
  });

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  );

  const handleGenerate = () => {
    const trimmed = jdText.trim();
    if (!trimmed || mutation.isPending) {
      return;
    }
    mutation.mutate(trimmed);
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) {
      return;
    }
    setSelection((items) => {
      const oldIndex = items.findIndex((item) => item.id === active.id);
      const newIndex = items.findIndex((item) => item.id === over.id);
      if (oldIndex < 0 || newIndex < 0) {
        return items;
      }
      return arrayMove(items, oldIndex, newIndex);
    });
  };

  const handleSelectionChange = (id: string, value: string) => {
    setSelection((items) =>
      items.map((item) => (item.id === id ? { ...item, text: value } : item)),
    );
  };

  const handleSelectionBlur = (id: string, value: string) => {
    const trimmed = value.trim();
    setSelection((items) =>
      items.map((item) => {
        if (item.id !== id) {
          return item;
        }
        if (!trimmed) {
          return { ...item, text: item.originalText || item.text };
        }
        return { ...item, text: trimmed };
      }),
    );
  };

  const handleSelectionDelete = (id: string) => {
    setSelection((items) => items.filter((item) => item.id !== id));
  };

  const handleAddExistingBullet = (option: BulletOption) => {
    setSelection((items) => {
      if (items.some((item) => item.id === option.id)) {
        return items;
      }
      return [
        ...items,
        {
          id: option.id,
          text: option.text,
          originalText: option.text,
          label: option.parentLabel,
          parentType: option.parentType,
          parentId: option.parentId,
          isTemp: false,
        },
      ];
    });
  };

  const toggleGroup = (key: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const nextTempId = () => {
    tempCounter.current += 1;
    return `tmp_${Date.now()}_${tempCounter.current}`;
  };

  const handleAddTempBullet = () => {
    const trimmed = newBulletText.trim();
    if (!trimmed) {
      setSelectionStatus({
        tone: "error",
        message: "Bullet text cannot be empty.",
      });
      return;
    }
    if (!newBulletParentId) {
      setSelectionStatus({
        tone: "error",
        message: "Select a parent experience or project first.",
      });
      return;
    }
    const tempId = nextTempId();
    const id = buildBulletId(newBulletType, newBulletParentId, tempId);
    const parentLabel =
      parentOptions.find((option) => option.id === newBulletParentId)?.label ||
      `${newBulletType} · ${newBulletParentId}`;

    setSelection((items) => [
      ...items,
      {
        id,
        text: trimmed,
        originalText: trimmed,
        label: parentLabel,
        parentType: newBulletType,
        parentId: newBulletParentId,
        tempId,
        isTemp: true,
      },
    ]);
    setNewBulletText("");
  };

  const buildRenderPayload = () => {
    const selected_ids = selection.map((item) => item.id);
    const additions = selection
      .filter((item) => item.isTemp)
      .map((item) => ({
        temp_id: item.tempId,
        parent_type: item.parentType as "experience" | "project",
        parent_id: item.parentId as string,
        text_latex: item.text.trim(),
      }))
      .filter((item) => item.text_latex);

    const edits: Record<string, string> = {};
    selection.forEach((item) => {
      if (item.isTemp) {
        return;
      }
      const trimmed = item.text.trim();
      if (!trimmed || trimmed === item.originalText) {
        return;
      }
      edits[item.id] = trimmed;
    });

    const temp_overrides: TempOverrides = {};
    if (additions.length) {
      temp_overrides.additions = additions;
    }
    if (Object.keys(edits).length) {
      temp_overrides.edits = edits;
    }

    return {
      selected_ids,
      temp_overrides: Object.keys(temp_overrides).length
        ? temp_overrides
        : undefined,
    };
  };

  const handleRender = () => {
    if (!runId) {
      return;
    }
    if (!selection.length) {
      setSelectionStatus({
        tone: "error",
        message: "Select at least one bullet before rendering.",
      });
      return;
    }
    const payload = buildRenderPayload();
    const invalidAddition = payload.temp_overrides?.additions?.find(
      (addition) =>
        !addition.parent_id || !addition.parent_type || !addition.text_latex,
    );
    if (invalidAddition) {
      setSelectionStatus({
        tone: "error",
        message: "Every temp bullet needs a parent and text.",
      });
      return;
    }
    renderMutation.mutate({ runId, payload });
  };

  const pdfUrl = result ? new URL(result.pdf_url, API_BASE_URL).toString() : "";

  const pendingEdits = useMemo(
    () =>
      selection.filter(
        (item) => !item.isTemp && item.text !== item.originalText,
      ).length,
    [selection],
  );

  const pendingAdditions = useMemo(
    () => selection.filter((item) => item.isTemp).length,
    [selection],
  );

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
          Paste a job description, run the agent loop, then adjust the selected
          bullets before rendering a final PDF.
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

      {result ? (
        <Card className="animate-rise">
          <CardHeader>
            <CardTitle>Selected bullets</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
              <span>
                Drag to reorder. Edit text inline, add custom bullets, or remove
                items before rendering.
              </span>
              <span>
                {selection.length} selected · {pendingEdits} edits · {pendingAdditions} additions
              </span>
            </div>
            {selectionStatus ? (
              <div
                className={cn(
                  "rounded-lg border px-3 py-2 text-sm",
                  selectionStatus.tone === "success"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                    : "border-destructive/30 bg-destructive/10 text-destructive",
                )}
              >
                {selectionStatus.message}
              </div>
            ) : null}

            {reportLoading ? (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                Loading selection...
              </div>
            ) : reportError ? (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
                Failed to load selection for this run.
              </div>
            ) : selection.length ? (
              <DndContext
                sensors={sensors}
                collisionDetection={closestCenter}
                onDragEnd={handleDragEnd}
              >
                <SortableContext
                  items={selection.map((item) => item.id)}
                  strategy={verticalListSortingStrategy}
                >
                  <div className="space-y-3">
                    {selection.map((item) => (
                      <SortableSelectionRow
                        key={item.id}
                        item={item}
                        onChange={handleSelectionChange}
                        onBlur={handleSelectionBlur}
                        onDelete={handleSelectionDelete}
                      />
                    ))}
                  </div>
                </SortableContext>
              </DndContext>
            ) : (
              <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                No bullets selected yet.
              </div>
            )}

            <div className="rounded-lg border bg-background p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="text-sm font-semibold">Bullet picker</div>
                <span className="text-xs text-muted-foreground">
                  {availableGroups.reduce(
                    (count, group) => count + group.items.length,
                    0,
                  )}{" "}
                  available
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                Add existing bullets that are not in the current selection.
              </p>
              <div className="mt-4 space-y-3">
                {availableGroups.length ? (
                  availableGroups.map((group) => {
                    const isCollapsed = collapsedGroups[group.key] ?? false;
                    return (
                      <div key={group.key} className="rounded-lg border">
                        <button
                          type="button"
                          className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left"
                          onClick={() => toggleGroup(group.key)}
                        >
                          <div className="flex items-center gap-2">
                            <ChevronDown
                              className={cn(
                                "h-4 w-4 transition-transform",
                                isCollapsed && "-rotate-90",
                              )}
                            />
                            <span className="text-sm font-medium">
                              {group.label}
                            </span>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {group.items.length} bullets
                          </span>
                        </button>
                        {!isCollapsed ? (
                          <div className="space-y-3 border-t px-3 py-3">
                            {group.items.map((item) => (
                              <div
                                key={item.id}
                                className="flex flex-col gap-2 rounded-lg border bg-muted/30 p-3 md:flex-row md:items-start md:justify-between"
                              >
                                <div className="text-sm text-muted-foreground">
                                  {item.text}
                                </div>
                                <Button
                                  type="button"
                                  variant="secondary"
                                  size="sm"
                                  onClick={() => handleAddExistingBullet(item)}
                                >
                                  <Plus className="h-4 w-4" />
                                  Add
                                </Button>
                              </div>
                            ))}
                          </div>
                        ) : null}
                      </div>
                    );
                  })
                ) : (
                  <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
                    All bullets are already selected.
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-lg border bg-muted/30 p-4">
              <div className="text-sm font-semibold">Add a temporary bullet</div>
              <div className="mt-3 grid gap-4 md:grid-cols-[160px_1fr]">
                <div className="space-y-2">
                  <Label htmlFor="temp-bullet-type">Section</Label>
                  <select
                    id="temp-bullet-type"
                    value={newBulletType}
                    onChange={(event) =>
                      setNewBulletType(
                        event.target.value as "experience" | "project",
                      )
                    }
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    <option value="experience">Experience</option>
                    <option value="project">Project</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="temp-bullet-parent">Parent</Label>
                  <select
                    id="temp-bullet-parent"
                    value={newBulletParentId}
                    onChange={(event) => setNewBulletParentId(event.target.value)}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  >
                    {parentOptions.map((option) => (
                      <option key={option.id} value={option.id}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  {!parentOptions.length ? (
                    <p className="text-xs text-muted-foreground">
                      Add a {newBulletType} in the editor first.
                    </p>
                  ) : null}
                </div>
              </div>
              <div className="mt-4 space-y-2">
                <Label htmlFor="temp-bullet-text">Bullet text</Label>
                <Textarea
                  id="temp-bullet-text"
                  value={newBulletText}
                  onChange={(event) => setNewBulletText(event.target.value)}
                  placeholder="Add a temporary bullet for this run..."
                />
              </div>
              <div className="mt-4 flex justify-end">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={handleAddTempBullet}
                  disabled={!parentOptions.length}
                >
                  <Plus className="h-4 w-4" />
                  Add bullet
                </Button>
              </div>
            </div>

            <div className="flex justify-end">
              <Button
                onClick={handleRender}
                disabled={renderMutation.isPending || !selection.length}
              >
                {renderMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Rendering
                  </>
                ) : (
                  "Render PDF"
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
