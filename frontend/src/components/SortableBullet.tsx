import { useEffect, useState } from "react";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { GripVertical } from "lucide-react";

import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { Bullet } from "@/types/schema";

type SortableBulletProps = {
  bullet: Bullet;
  onUpdate: (next: Bullet) => void | Promise<void>;
};

export function SortableBullet({ bullet, onUpdate }: SortableBulletProps) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: bullet.id });
  const [value, setValue] = useState(bullet.text_latex);

  useEffect(() => {
    setValue(bullet.text_latex);
  }, [bullet.text_latex]);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  const handleBlur = () => {
    const next = value.trim();
    if (!next) {
      setValue(bullet.text_latex);
      return;
    }
    if (next === bullet.text_latex) {
      return;
    }
    void onUpdate({ ...bullet, text_latex: next });
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={cn(
        "flex items-start gap-2 rounded-lg border bg-card p-3 shadow-sm",
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
      <Textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onBlur={handleBlur}
        className="min-h-[84px]"
      />
    </div>
  );
}
