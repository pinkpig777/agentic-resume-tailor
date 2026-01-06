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
import type { Bullet, Project, ProjectUpdatePayload } from "@/types/schema";

const REQUIRED_FIELDS: Array<keyof ProjectUpdatePayload> = ["name"];

type ProjectCardProps = {
  project: Project;
  onProjectUpdate: (
    projectId: string,
    payload: ProjectUpdatePayload,
  ) => void | Promise<void>;
  onProjectDelete: (projectId: string) => void | Promise<void>;
  onBulletCreate: (projectId: string, text: string) => void | Promise<void>;
  onBulletUpdate: (projectId: string, bullet: Bullet) => void | Promise<void>;
  onBulletDelete: (projectId: string, bulletId: string) => void | Promise<void>;
  onBulletsReorder?: (projectId: string, bullets: Bullet[]) => void | Promise<void>;
  collapsed?: boolean;
  onToggle?: () => void;
};

type ProjectDraft = {
  name: string;
  technologies: string;
};

const buildDraft = (project: Project): ProjectDraft => ({
  name: project.name,
  technologies: project.technologies,
});

export function ProjectCard({
  project,
  onProjectUpdate,
  onProjectDelete,
  onBulletCreate,
  onBulletUpdate,
  onBulletDelete,
  onBulletsReorder,
  collapsed = false,
  onToggle,
}: ProjectCardProps) {
  const [items, setItems] = useState<Bullet[]>(project.bullets);
  const [draft, setDraft] = useState<ProjectDraft>(() => buildDraft(project));
  const [newBullet, setNewBullet] = useState("");
  const contentId = `project-${project.project_id}-content`;

  useEffect(() => {
    setItems(project.bullets);
  }, [project.bullets]);

  useEffect(() => {
    setDraft(buildDraft(project));
  }, [project.name, project.technologies]);

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
      void onBulletsReorder(project.project_id, reordered);
    } else {
      reordered.forEach((bullet) =>
        void onBulletUpdate(project.project_id, bullet),
      );
    }
  };

  const handleFieldBlur = (field: keyof ProjectDraft) => {
    const next = draft[field].trim();
    const current = project[field];
    if (REQUIRED_FIELDS.includes(field) && !next) {
      setDraft((prev) => ({ ...prev, [field]: current }));
      return;
    }
    if (next === current) {
      return;
    }
    void onProjectUpdate(project.project_id, { [field]: next });
  };

  const handleAddBullet = () => {
    const next = newBullet.trim();
    if (!next) {
      return;
    }
    void onBulletCreate(project.project_id, next);
    setNewBullet("");
  };

  return (
    <Card>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <CardTitle className="text-base">
              {project.name || "Project"}
            </CardTitle>
            <div className="text-xs text-muted-foreground">
              Project ID: {project.project_id}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggle}
              aria-label="Toggle project details"
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
              onClick={() => onProjectDelete(project.project_id)}
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
              <Label htmlFor={`${project.project_id}-name`}>Name</Label>
              <Input
                id={`${project.project_id}-name`}
                value={draft.name}
                onChange={(event) =>
                  setDraft((prev) => ({ ...prev, name: event.target.value }))
                }
                onBlur={() => handleFieldBlur("name")}
                placeholder="Project name"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor={`${project.project_id}-tech`}>Technologies</Label>
              <Input
                id={`${project.project_id}-tech`}
                value={draft.technologies}
                onChange={(event) =>
                  setDraft((prev) => ({
                    ...prev,
                    technologies: event.target.value,
                  }))
                }
                onBlur={() => handleFieldBlur("technologies")}
                placeholder="React, FastAPI, Postgres"
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
                          onBulletUpdate(project.project_id, next)
                        }
                        onDelete={(id) => onBulletDelete(project.project_id, id)}
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
            <Label htmlFor={`${project.project_id}-new-bullet`}>Add bullet</Label>
            <Textarea
              id={`${project.project_id}-new-bullet`}
              value={newBullet}
              onChange={(event) => setNewBullet(event.target.value)}
              placeholder="Add a new project highlight..."
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
