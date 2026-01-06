import { useEffect, useMemo, useState } from "react";
import { Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { Education, EducationUpdatePayload } from "@/types/schema";

const parseBullets = (value: string) =>
  value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

type EducationCardProps = {
  education: Education;
  onUpdate: (
    educationId: number,
    payload: EducationUpdatePayload,
  ) => void | Promise<void>;
  onDelete: (educationId: number) => void | Promise<void>;
};

type EducationDraft = {
  school: string;
  degree: string;
  dates: string;
  location: string;
  bulletsText: string;
};

const buildDraft = (education: Education): EducationDraft => ({
  school: education.school,
  degree: education.degree,
  dates: education.dates,
  location: education.location,
  bulletsText: education.bullets.join("\n"),
});

export function EducationCard({ education, onUpdate, onDelete }: EducationCardProps) {
  const [draft, setDraft] = useState<EducationDraft>(() => buildDraft(education));

  useEffect(() => {
    setDraft(buildDraft(education));
  }, [education.school, education.degree, education.dates, education.location, education.bullets]);

  const normalizedBullets = useMemo(
    () => parseBullets(draft.bulletsText),
    [draft.bulletsText],
  );

  const handleFieldBlur = (field: keyof EducationDraft) => {
    if (field === "bulletsText") {
      const current = education.bullets;
      const next = normalizedBullets;
      if (current.join("\n") === next.join("\n")) {
        return;
      }
      void onUpdate(education.id, { bullets: next });
      return;
    }

    const next = draft[field].trim();
    const current = education[field];
    if (field === "school" && !next) {
      setDraft((prev) => ({ ...prev, school: current }));
      return;
    }
    if (next === current) {
      return;
    }
    void onUpdate(education.id, { [field]: next } as EducationUpdatePayload);
  };

  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="text-base">
              {education.school || "Education"}
            </CardTitle>
            <div className="text-xs text-muted-foreground">ID: {education.id}</div>
          </div>
          <Button variant="destructive" size="sm" onClick={() => onDelete(education.id)}>
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor={`edu-${education.id}-school`}>School</Label>
            <Input
              id={`edu-${education.id}-school`}
              value={draft.school}
              onChange={(event) =>
                setDraft((prev) => ({ ...prev, school: event.target.value }))
              }
              onBlur={() => handleFieldBlur("school")}
              placeholder="University name"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`edu-${education.id}-degree`}>Degree</Label>
            <Input
              id={`edu-${education.id}-degree`}
              value={draft.degree}
              onChange={(event) =>
                setDraft((prev) => ({ ...prev, degree: event.target.value }))
              }
              onBlur={() => handleFieldBlur("degree")}
              placeholder="B.S. Computer Science"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`edu-${education.id}-dates`}>Dates</Label>
            <Input
              id={`edu-${education.id}-dates`}
              value={draft.dates}
              onChange={(event) =>
                setDraft((prev) => ({ ...prev, dates: event.target.value }))
              }
              onBlur={() => handleFieldBlur("dates")}
              placeholder="2016 - 2020"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={`edu-${education.id}-location`}>Location</Label>
            <Input
              id={`edu-${education.id}-location`}
              value={draft.location}
              onChange={(event) =>
                setDraft((prev) => ({ ...prev, location: event.target.value }))
              }
              onBlur={() => handleFieldBlur("location")}
              placeholder="City, Country"
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor={`edu-${education.id}-bullets`}>Highlights</Label>
          <Textarea
            id={`edu-${education.id}-bullets`}
            value={draft.bulletsText}
            onChange={(event) =>
              setDraft((prev) => ({ ...prev, bulletsText: event.target.value }))
            }
            onBlur={() => handleFieldBlur("bulletsText")}
            placeholder="One highlight per line"
          />
          <p className="text-xs text-muted-foreground">
            One bullet per line. Changes save on blur.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
