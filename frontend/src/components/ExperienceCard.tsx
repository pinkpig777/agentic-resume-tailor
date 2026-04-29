import { useEffect, useState } from "react";
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
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { ChevronDown, Plus, Trash2 } from "lucide-react";

import { SortableBullet } from "@/components/SortableBullet";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { Bullet, Experience, ExperienceUpdatePayload } from "@/types/schema";

const REQUIRED_FIELDS: Array<keyof ExperienceUpdatePayload> = ["company", "role"];

type ExperienceCardProps = {
  experience: Experience;
  onExperienceUpdate: (
    jobId: string,
    payload: ExperienceUpdatePayload,
  ) => void | Promise<void>;
  onExperienceDelete: (jobId: string) => void | Promise<void>;
  onBulletCreate: (jobId: string, text: string) => void | Promise<void>;
  onBulletUpdate: (jobId: string, bullet: Bullet) => void | Promise<void>;
  onBulletDelete: (jobId: string, bulletId: string) => void | Promise<void>;
  onBulletsReorder?: (jobId: string, bullets: Bullet[]) => void | Promise<void>;
  collapsed?: boolean;
  onToggle?: () => void;
};

type ExperienceDraft = {
  company: string;
  role: string;
  dates: string;
  location: string;
};

const buildDraft = (experience: Experience): ExperienceDraft => ({
  company: experience.company,
  role: experience.role,
  dates: experience.dates,
  location: experience.location,
});

export function ExperienceCard({
  experience,
  onExperienceUpdate,
  onExperienceDelete,
  onBulletCreate,
  onBulletUpdate,
  onBulletDelete,
  onBulletsReorder,
  collapsed = false,
  onToggle,
}: ExperienceCardProps) {
  const [items, setItems] = useState<Bullet[]>(experience.bullets);
  const [draft, setDraft] = useState<ExperienceDraft>(() => buildDraft(experience));
  const [newBullet, setNewBullet] = useState("");
  const contentId = `experience-${experience.job_id}-content`;

  useEffect(() => {
    setItems(experience.bullets);
  }, [experience.bullets]);

  useEffect(() => {
    setDraft(buildDraft(experience));
  }, [experience.company, experience.role, experience.dates, experience.location]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) {
      return;
    }

    const activeId = String(active.id);
    const overId = String(over.id);
    const oldIndex = items.findIndex((item) => item.id === activeId);
    const newIndex = items.findIndex((item) => item.id === overId);
    if (oldIndex < 0 || newIndex < 0) {
      return;
    }

    const reordered = arrayMove(items, oldIndex, newIndex).map((bullet, index) => ({
      ...bullet,
      sort_order: index + 1,
    }));

    setItems(reordered);

    if (onBulletsReorder) {
      void onBulletsReorder(experience.job_id, reordered);
    } else {
      reordered.forEach((bullet) => void onBulletUpdate(experience.job_id, bullet));
    }
  };

  const handleFieldBlur = (field: keyof ExperienceDraft) => {
    const next = draft[field].trim();
    const current = experience[field];
    if (REQUIRED_FIELDS.includes(field) && !next) {
      setDraft((prev) => ({ ...prev, [field]: current }));
      return;
    }
    if (next === current) {
      return;
    }
    void onExperienceUpdate(experience.job_id, { [field]: next });
  };

  const handleAddBullet = () => {
    const next = newBullet.trim();
    if (!next) {
      return;
    }
    void onBulletCreate(experience.job_id, next);
    setNewBullet("");
  };

  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="text-base">
              {experience.role || "Role"} · {experience.company || "Company"}
            </CardTitle>
            <div className="text-xs text-muted-foreground">
              Job ID: {experience.job_id}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggle}
              aria-label="Toggle experience details"
              aria-expanded={!collapsed}
              aria-controls={contentId}
            >
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform",
                  collapsed && "-rotate-90",
                )}
              />
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={() => onExperienceDelete(experience.job_id)}
            >
              <Trash2 className="h-4 w-4" />
              Delete
            </Button>
          </div>
        </div>
      </CardHeader>
      {collapsed ? null : (
        <CardContent id={contentId} className="space-y-5">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor={`${experience.job_id}-company`}>Company</Label>
              <Input
                id={`${experience.job_id}-company`}
                value={draft.company}
                onChange={(event) =>
                  setDraft((prev) => ({ ...prev, company: event.target.value }))
                }
                onBlur={() => handleFieldBlur("company")}
                placeholder="Company name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor={`${experience.job_id}-role`}>Role</Label>
              <Input
                id={`${experience.job_id}-role`}
                value={draft.role}
                onChange={(event) =>
                  setDraft((prev) => ({ ...prev, role: event.target.value }))
                }
                onBlur={() => handleFieldBlur("role")}
                placeholder="Role title"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor={`${experience.job_id}-dates`}>Dates</Label>
              <Input
                id={`${experience.job_id}-dates`}
                value={draft.dates}
                onChange={(event) =>
                  setDraft((prev) => ({ ...prev, dates: event.target.value }))
                }
                onBlur={() => handleFieldBlur("dates")}
                placeholder="2021 - Present"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor={`${experience.job_id}-location`}>Location</Label>
              <Input
                id={`${experience.job_id}-location`}
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
            <Label>Bullets</Label>
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={items.map((bullet) => bullet.id)}
                strategy={verticalListSortingStrategy}
              >
                <div className="space-y-2">
                  {items.length ? (
                    items.map((bullet) => (
                      <SortableBullet
                        key={bullet.id}
                        bullet={bullet}
                        onUpdate={(next) =>
                          onBulletUpdate(experience.job_id, next)
                        }
                        onDelete={(id) => onBulletDelete(experience.job_id, id)}
                      />
                    ))
                  ) : (
                    <div className="rounded-lg border border-dashed p-3 text-xs text-muted-foreground">
                      No bullets yet. Add one below.
                    </div>
                  )}
                </div>
              </SortableContext>
            </DndContext>
          </div>

          <div className="space-y-2">
            <Label htmlFor={`${experience.job_id}-new-bullet`}>Add bullet</Label>
            <Textarea
              id={`${experience.job_id}-new-bullet`}
              value={newBullet}
              onChange={(event) => setNewBullet(event.target.value)}
              placeholder="Add a new impact statement..."
            />
            <div className="flex justify-end">
              <Button
                type="button"
                variant="secondary"
                size="sm"
                onClick={handleAddBullet}
              >
                <Plus className="h-4 w-4" />
                Add bullet
              </Button>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
