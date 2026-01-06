import { FileText, GripVertical, Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function EditorPage() {
  return (
    <div className="space-y-6">
      <header className="space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
          <FileText className="h-4 w-4" />
          Resume Editor
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">
          Keep your profile sharp.
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          This page will host the drag-and-drop experience editor and bullet
          library once the data layer is wired in.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-2">
        <Card className="animate-rise">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <GripVertical className="h-4 w-4 text-muted-foreground" />
              Drag and reorder
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            We will map experiences to sortable bullet lists with autosave on
            blur and instant re-ingest triggers.
          </CardContent>
        </Card>
        <Card className="animate-rise animate-rise-delay-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-muted-foreground" />
              Highlight impact
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Sync your most relevant bullets and keep them ready for the next
            generation run.
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
