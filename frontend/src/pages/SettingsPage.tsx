import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, RefreshCcw, SlidersHorizontal } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { fetchSettings, updateSettings } from "@/lib/api";
import type { SettingsData } from "@/types/schema";

const booleanFields = [
  "auto_reingest_on_save",
  "use_jd_parser",
  "skip_pdf",
  "log_json",
] as const;

const integerFields = [
  "max_bullets",
  "per_query_k",
  "final_k",
  "max_iters",
  "threshold",
  "boost_top_n_missing",
  "port",
] as const;

const floatFields = [
  "alpha",
  "must_weight",
  "quant_bonus_per_hit",
  "quant_bonus_cap",
  "boost_weight",
  "experience_weight",
] as const;

const numberFields = [...integerFields, ...floatFields] as const;

const logLevelOptions = ["DEBUG", "INFO", "WARNING", "ERROR"];
const jdModelOptions = [
  "gpt-4.1-nano-2025-04-14",
  "gpt-4.1-mini-2025-04-14",
  "gpt-4.1-2025-04-14",
  "gpt-4o-mini-2024-07-18",
  "gpt-4o-2024-08-06",
];

const tooltips: Record<string, string> = {
  config_path: "Resolved user settings file (read-only).",
  db_path: "Directory for the Chroma vector store.",
  sql_db_url: "SQLAlchemy URL for the resume database.",
  export_file: "Path to the exported resume JSON.",
  output_pdf_name: "Optional filename for rendered PDF downloads.",
  template_dir: "Directory containing LaTeX templates.",
  output_dir: "Directory where PDF/TeX/report artifacts are written.",
  canon_config: "Canonicalization rules JSON.",
  family_config: "Skill family mapping JSON.",
  max_bullets: "Max bullets allowed in the final resume.",
  per_query_k: "Number of candidates retrieved per query.",
  final_k: "Number of candidates scored per run.",
  max_iters: "Max iterations for the agent loop.",
  threshold: "Minimum hybrid score to accept bullets.",
  alpha: "Blend factor between retrieval and coverage.",
  must_weight: "Weight applied to must-have keywords.",
  quant_bonus_per_hit: "Bonus per quantified keyword hit.",
  quant_bonus_cap: "Maximum quant bonus.",
  boost_weight: "Boost weight for missing keywords.",
  boost_top_n_missing: "Number of missing keywords boosted.",
  experience_weight: "Multiplier for experience bullets vs projects.",
  collection_name: "Chroma collection name used for retrieval.",
  embed_model: "Embedding model ID used to embed bullets.",
  jd_model: "OpenAI model used by the JD parser.",
  api_url: "Backend API base URL.",
  cors_origins: "Allowed origins for CORS requests.",
  port: "Port for the FastAPI server.",
  run_id: "Optional fixed run ID; overwrites output files.",
  auto_reingest_on_save: "Automatically re-ingest after saving edits.",
  use_jd_parser: "Parse job descriptions before retrieval.",
  skip_pdf: "Skip PDF rendering and only produce TeX.",
  log_level: "Logging verbosity.",
  log_json: "Emit JSON formatted logs.",
};

type BooleanField = (typeof booleanFields)[number];
type IntegerField = (typeof integerFields)[number];
type NumberField = (typeof numberFields)[number];
type TextField = Exclude<
  keyof SettingsData,
  BooleanField | NumberField | "config_path"
>;

type SettingsFormState = {
  [K in BooleanField]: boolean;
} & {
  [K in NumberField]: string;
} & {
  [K in TextField]: string;
} & {
  config_path: string;
};

type StatusTone = "success" | "error";

type StatusMessage = {
  tone: StatusTone;
  message: string;
};

const buildFormState = (settings: SettingsData): SettingsFormState => {
  const base: Record<string, string | boolean | number | null> = {
    ...settings,
    run_id: settings.run_id ?? "",
    output_pdf_name: settings.output_pdf_name ?? "",
  };

  numberFields.forEach((field) => {
    base[field] = settings[field] === null ? "" : String(settings[field]);
  });

  booleanFields.forEach((field) => {
    base[field] = Boolean(settings[field]);
  });

  base.config_path = settings.config_path;

  return base as SettingsFormState;
};

const parseNumber = (field: NumberField, raw: string) => {
  if (integerFields.includes(field as IntegerField)) {
    return Number.parseInt(raw, 10);
  }
  return Number.parseFloat(raw);
};

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const [draft, setDraft] = useState<SettingsFormState | null>(null);
  const [status, setStatus] = useState<StatusMessage | null>(null);

  const {
    data: settings,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["settings"],
    queryFn: fetchSettings,
  });

  useEffect(() => {
    if (settings) {
      setDraft(buildFormState(settings));
    }
  }, [settings]);

  useEffect(() => {
    if (!status || status.tone === "error") {
      return undefined;
    }
    const timer = window.setTimeout(() => setStatus(null), 1800);
    return () => window.clearTimeout(timer);
  }, [status]);

  const updateMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: (updated) => {
      queryClient.setQueryData(["settings"], updated);
      setDraft(buildFormState(updated));
      setStatus({ tone: "success", message: "Settings saved." });
    },
    onError: () => {
      setStatus({ tone: "error", message: "Failed to update settings." });
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
  });

  const handleTextBlur = (field: TextField, rawValue?: string) => {
    if (!settings || !draft) {
      return;
    }
    const currentValue = rawValue ?? draft[field];
    const next = currentValue.trim();
    const current = (settings[field] ?? "") as string | null;
    const normalizedCurrent = current ?? "";
    if (next === normalizedCurrent) {
      return;
    }
    const optionalNull = field === "run_id" || field === "output_pdf_name";
    const payloadValue = optionalNull && !next ? null : next;
    updateMutation.mutate({ [field]: payloadValue });
  };

  const handleNumberBlur = (field: NumberField, rawValue?: string) => {
    if (!settings || !draft) {
      return;
    }
    const raw = (rawValue ?? draft[field]).trim();
    if (!raw) {
      setDraft((prev) =>
        prev
          ? { ...prev, [field]: String(settings[field]) }
          : prev,
      );
      return;
    }
    const parsed = parseNumber(field, raw);
    if (Number.isNaN(parsed)) {
      setDraft((prev) =>
        prev
          ? { ...prev, [field]: String(settings[field]) }
          : prev,
      );
      return;
    }
    if (parsed === settings[field]) {
      return;
    }
    updateMutation.mutate({ [field]: parsed });
  };

  const handleToggle = (field: BooleanField) => {
    if (!settings || !draft) {
      return;
    }
    const next = !draft[field];
    setDraft((prev) => (prev ? { ...prev, [field]: next } : prev));
    if (next === settings[field]) {
      return;
    }
    updateMutation.mutate({ [field]: next });
  };

  const refreshSettings = () => {
    queryClient.invalidateQueries({ queryKey: ["settings"] });
  };

  const renderTextField = (
    field: TextField,
    label: string,
    placeholder?: string,
    description?: string,
    tooltip?: string,
  ) => {
    if (!draft) {
      return null;
    }
    const id = `settings-${field}`;
    const helpText = tooltip ?? description ?? label;
    return (
      <div className="space-y-2" key={field}>
        <Label htmlFor={id} title={helpText}>
          {label}
        </Label>
        <Input
          id={id}
          value={draft[field]}
          onChange={(event) =>
            setDraft((prev) =>
              prev ? { ...prev, [field]: event.target.value } : prev,
            )
          }
          onBlur={(event) => handleTextBlur(field, event.currentTarget.value)}
          placeholder={placeholder}
          title={helpText}
        />
        {description ? (
          <p className="text-xs text-muted-foreground">{description}</p>
        ) : null}
      </div>
    );
  };

  const renderNumberField = (
    field: NumberField,
    label: string,
    placeholder?: string,
    description?: string,
    tooltip?: string,
  ) => {
    if (!draft) {
      return null;
    }
    const id = `settings-${field}`;
    const helpText = tooltip ?? description ?? label;
    return (
      <div className="space-y-2" key={field}>
        <Label htmlFor={id} title={helpText}>
          {label}
        </Label>
        <Input
          id={id}
          type="number"
          value={draft[field]}
          onChange={(event) =>
            setDraft((prev) =>
              prev ? { ...prev, [field]: event.target.value } : prev,
            )
          }
          onBlur={(event) => handleNumberBlur(field, event.currentTarget.value)}
          placeholder={placeholder}
          title={helpText}
        />
        {description ? (
          <p className="text-xs text-muted-foreground">{description}</p>
        ) : null}
      </div>
    );
  };

  const renderToggleField = (
    field: BooleanField,
    label: string,
    description?: string,
    tooltip?: string,
  ) => {
    if (!draft) {
      return null;
    }
    const id = `settings-${field}`;
    const helpText = tooltip ?? description ?? label;
    return (
      <div className="flex items-start justify-between gap-3" key={field}>
        <div className="space-y-1">
          <Label
            htmlFor={id}
            title={helpText}
            className="text-sm normal-case tracking-normal"
          >
            {label}
          </Label>
          {description ? (
            <p className="text-xs text-muted-foreground">{description}</p>
          ) : null}
        </div>
        <input
          id={id}
          type="checkbox"
          checked={draft[field]}
          onChange={() => handleToggle(field)}
          title={helpText}
          className="mt-1 h-4 w-4 rounded border-input text-primary focus:ring-primary"
        />
      </div>
    );
  };

  const statusTone = status?.tone ?? "success";
  const isCustomJdModel = draft
    ? !jdModelOptions.includes(draft.jd_model)
    : false;
  const jdModelSelectValue = isCustomJdModel
    ? "__custom__"
    : draft?.jd_model ?? jdModelOptions[0];

  const handleJdModelSelect = (value: string) => {
    if (!draft) {
      return;
    }
    if (value === "__custom__") {
      setDraft((prev) => {
        if (!prev) {
          return prev;
        }
        if (jdModelOptions.includes(prev.jd_model)) {
          return { ...prev, jd_model: "" };
        }
        return prev;
      });
      return;
    }
    setDraft((prev) => (prev ? { ...prev, jd_model: value } : prev));
    if (settings && value !== settings.jd_model) {
      updateMutation.mutate({ jd_model: value });
    }
  };

  const summary = useMemo(
    () => ({
      requestStatus: updateMutation.isPending ? "Saving" : "Autosave",
      statusMessage: status?.message ?? "Changes save on blur.",
    }),
    [status, updateMutation.isPending],
  );

  if (isLoading || !draft) {
    return (
      <div className="space-y-6">
        <header className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            <SlidersHorizontal className="h-4 w-4" />
            Settings
          </div>
          <h1 className="text-3xl font-semibold md:text-4xl">Tune the pipeline.</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Loading settings from the FastAPI config file.
          </p>
        </header>
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            Loading settings...
          </CardContent>
        </Card>
      </div>
    );
  }

  if (isError || !settings) {
    return (
      <div className="space-y-6">
        <header className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
            <SlidersHorizontal className="h-4 w-4" />
            Settings
          </div>
          <h1 className="text-3xl font-semibold md:text-4xl">Tune the pipeline.</h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            The settings view could not reach the API.
          </p>
        </header>
        <Card>
          <CardContent className="py-10 text-center text-sm text-destructive">
            Failed to load settings.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <SlidersHorizontal className="h-4 w-4" />
          Settings
        </div>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold md:text-4xl">
              Tune the pipeline.
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Adjust defaults for scoring, retrieval, and export behavior.
            </p>
          </div>
          <Button variant="secondary" onClick={refreshSettings}>
            <RefreshCcw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
        <div
          className={cn(
            "flex items-center gap-2 rounded-lg border px-3 py-2 text-sm",
            statusTone === "error"
              ? "border-destructive/30 bg-destructive/10 text-destructive"
              : "border-emerald-200 bg-emerald-50 text-emerald-700",
          )}
        >
          {updateMutation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : null}
          <span className="font-semibold uppercase text-[10px] tracking-[0.18em]">
            {summary.requestStatus}
          </span>
          <span>{summary.statusMessage}</span>
        </div>
      </header>

      <section className="space-y-4">
        <Card className="animate-rise">
          <CardHeader>
            <CardTitle>Config file</CardTitle>
            <CardDescription>
              Stored user overrides are saved to this JSON file.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Label htmlFor="settings-config-path" title={tooltips.config_path}>
              Config path
            </Label>
            <Input
              id="settings-config-path"
              value={draft.config_path}
              readOnly
              title={tooltips.config_path}
            />
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Paths & storage</CardTitle>
            <CardDescription>
              Control where data is stored and exported on disk.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            {renderTextField(
              "db_path",
              "Chroma DB path",
              "data/processed/chroma_db",
              undefined,
              tooltips.db_path,
            )}
            {renderTextField(
              "sql_db_url",
              "SQL DB URL",
              "sqlite:///data/processed/resume.db",
              undefined,
              tooltips.sql_db_url,
            )}
            {renderTextField(
              "export_file",
              "Export JSON file",
              "data/my_experience.json",
              undefined,
              tooltips.export_file,
            )}
            {renderTextField(
              "output_pdf_name",
              "Output PDF name",
              "tailored_resume.pdf",
              "Optional download filename for rendered PDFs.",
              tooltips.output_pdf_name,
            )}
            {renderTextField(
              "template_dir",
              "Template dir",
              "templates",
              undefined,
              tooltips.template_dir,
            )}
            {renderTextField(
              "output_dir",
              "Output dir",
              "output",
              undefined,
              tooltips.output_dir,
            )}
            {renderTextField(
              "canon_config",
              "Canonicalization config",
              "config/canonicalization.json",
              undefined,
              tooltips.canon_config,
            )}
            {renderTextField(
              "family_config",
              "Families config",
              "config/families.json",
              undefined,
              tooltips.family_config,
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generation defaults</CardTitle>
            <CardDescription>
              Settings that shape the scoring loop and selection behavior.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            {renderNumberField(
              "max_bullets",
              "Max bullets",
              "16",
              "Total bullets allowed in the final resume.",
              tooltips.max_bullets,
            )}
            {renderNumberField(
              "per_query_k",
              "Per query K",
              "10",
              "Candidates fetched per query.",
              tooltips.per_query_k,
            )}
            {renderNumberField(
              "final_k",
              "Final K",
              "30",
              "Candidates evaluated in the final pool.",
              tooltips.final_k,
            )}
            {renderNumberField(
              "max_iters",
              "Max iterations",
              "3",
              "Number of agent loop passes.",
              tooltips.max_iters,
            )}
            {renderNumberField(
              "threshold",
              "Threshold",
              "80",
              "Minimum hybrid score for acceptance.",
              tooltips.threshold,
            )}
            {renderNumberField(
              "alpha",
              "Alpha",
              "0.7",
              "Blend factor for scoring.",
              tooltips.alpha,
            )}
            {renderNumberField(
              "must_weight",
              "Must weight",
              "0.8",
              "Weight applied to must-have keywords.",
              tooltips.must_weight,
            )}
            {renderNumberField(
              "quant_bonus_per_hit",
              "Quant bonus per hit",
              "0.05",
              "Bonus per quantified keyword hit.",
              tooltips.quant_bonus_per_hit,
            )}
            {renderNumberField(
              "quant_bonus_cap",
              "Quant bonus cap",
              "0.2",
              "Maximum bonus from quantified hits.",
              tooltips.quant_bonus_cap,
            )}
            {renderNumberField(
              "boost_weight",
              "Boost weight",
              "1.6",
              "Multiplier for missing keywords.",
              tooltips.boost_weight,
            )}
            {renderNumberField(
              "experience_weight",
              "Experience weight",
              "1.2",
              "Prefer experience bullets over projects.",
              tooltips.experience_weight,
            )}
            {renderNumberField(
              "boost_top_n_missing",
              "Boost top N",
              "6",
              "Count of missing keywords to boost.",
              tooltips.boost_top_n_missing,
            )}
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Retrieval & runtime</CardTitle>
            <CardDescription>
              Tune embeddings, collection, and runtime defaults.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
              {renderTextField(
                "collection_name",
                "Chroma collection",
                "resume_experience",
                undefined,
                tooltips.collection_name,
              )}
              {renderTextField(
                "embed_model",
                "Embedding model",
                "BAAI/bge-small-en-v1.5",
                undefined,
                tooltips.embed_model,
              )}
              <div className="space-y-2">
                <Label htmlFor="settings-jd-model" title={tooltips.jd_model}>
                  JD model
                </Label>
                <select
                  id="settings-jd-model"
                  value={jdModelSelectValue}
                  onChange={(event) => handleJdModelSelect(event.target.value)}
                  title={tooltips.jd_model}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  {jdModelOptions.map((model) => (
                    <option value={model} key={model}>
                      {model}
                    </option>
                  ))}
                  <option value="__custom__">Custom…</option>
                </select>
                {isCustomJdModel ? (
                  <Input
                    value={draft.jd_model}
                    onChange={(event) =>
                      setDraft((prev) =>
                        prev ? { ...prev, jd_model: event.target.value } : prev,
                      )
                    }
                    onBlur={(event) =>
                      handleTextBlur("jd_model", event.currentTarget.value)
                    }
                    placeholder="Enter a model id"
                    title={tooltips.jd_model}
                  />
                ) : null}
              </div>
              {renderTextField(
                "api_url",
                "API URL",
                "http://localhost:8000",
                undefined,
                tooltips.api_url,
              )}
              {renderTextField(
                "cors_origins",
                "CORS origins",
                "*",
                "Comma-delimited origins or * for all.",
                tooltips.cors_origins,
              )}
              {renderNumberField(
                "port",
                "API port",
                "8000",
                undefined,
                tooltips.port,
              )}
              {renderTextField(
                "run_id",
                "Pinned run ID",
                "Optional",
                "Leave blank to auto-generate per run.",
                tooltips.run_id,
              )}
            </div>
            <div className="space-y-4">
              {renderToggleField(
                "auto_reingest_on_save",
                "Auto re-ingest on save",
                "Rebuild Chroma whenever profile data changes.",
                tooltips.auto_reingest_on_save,
              )}
              {renderToggleField(
                "use_jd_parser",
                "Use JD parser",
                "Parse the job description before retrieval.",
                tooltips.use_jd_parser,
              )}
              {renderToggleField(
                "skip_pdf",
                "Skip PDF rendering",
                "Generate LaTeX only.",
                tooltips.skip_pdf,
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Logging</CardTitle>
            <CardDescription>
              Adjust verbosity for local debugging.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="settings-log-level" title={tooltips.log_level}>
                Log level
              </Label>
              <Input
                id="settings-log-level"
                list="log-levels"
                value={draft.log_level}
                onChange={(event) =>
                  setDraft((prev) =>
                    prev ? { ...prev, log_level: event.target.value } : prev,
                  )
                }
                onBlur={(event) =>
                  handleTextBlur("log_level", event.currentTarget.value)
                }
                placeholder="INFO"
                title={tooltips.log_level}
              />
              <datalist id="log-levels">
                {logLevelOptions.map((level) => (
                  <option value={level} key={level} />
                ))}
              </datalist>
            </div>
            <div className="space-y-2">
              {renderToggleField(
                "log_json",
                "JSON logging",
                "Emit structured logs for collectors.",
                tooltips.log_json,
              )}
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
