import { useEffect, useState } from "react";
import {
  DndContext,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  arrayMove,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";

import { SortableBullet } from "@/components/SortableBullet";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Bullet, Experience } from "@/types/schema";

type ExperienceCardProps = {
  experience: Experience;
  onBulletUpdate: (jobId: string, bullet: Bullet) => void | Promise<void>;
  onBulletsReorder?: (jobId: string, bullets: Bullet[]) => void | Promise<void>;
};

export function ExperienceCard({
  experience,
  onBulletUpdate,
  onBulletsReorder,
}: ExperienceCardProps) {
  const [items, setItems] = useState<Bullet[]>(experience.bullets);

  useEffect(() => {
    setItems(experience.bullets);
  }, [experience.bullets]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
  );

  const handleDragEnd = (event: { active: { id: string }; over?: { id: string } | null }) => {
    const { active, over } = event;
    if (!over || active.id === over.id) {
      return;
    }

    const oldIndex = items.findIndex((item) => item.id === active.id);
    const newIndex = items.findIndex((item) => item.id === over.id);
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

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {experience.role} · {experience.company}
        </CardTitle>
        <div className="text-sm text-muted-foreground">
          {experience.dates} · {experience.location}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
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
              {items.map((bullet) => (
                <SortableBullet
                  key={bullet.id}
                  bullet={bullet}
                  onUpdate={(next) => onBulletUpdate(experience.job_id, next)}
                />
              ))}
            </div>
          </SortableContext>
        </DndContext>
      </CardContent>
    </Card>
  );
}
