import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { FileText, Settings, Sparkles } from "lucide-react";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";

import { cn } from "@/lib/utils";
import EditorPage from "@/pages/EditorPage";
import GeneratePage from "@/pages/GeneratePage";
import SettingsPage from "@/pages/SettingsPage";

const queryClient = new QueryClient();

const navItems = [
  { to: "/editor", label: "Editor", icon: FileText },
  { to: "/generate", label: "Generate", icon: Sparkles },
  { to: "/settings", label: "Settings", icon: Settings },
];

function AppShell() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-6 md:flex-row">
          <aside className="w-full md:w-64">
            <div className="rounded-2xl border bg-white/80 p-4 shadow-sm backdrop-blur">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.25em] text-muted-foreground">
                    ART
                  </div>
                  <div className="text-lg font-semibold">Agentic Resume Tailor</div>
                </div>
                <span className="rounded-full bg-secondary px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-secondary-foreground">
                  Local
                </span>
              </div>

              <nav className="mt-6 space-y-1">
                {navItems.map(({ to, label, icon: Icon }) => (
                  <NavLink
                    key={to}
                    to={to}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition",
                        isActive
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
                      )
                    }
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </NavLink>
                ))}
              </nav>

              <div className="mt-6 rounded-lg border bg-background/80 p-3 text-xs text-muted-foreground">
                FastAPI: <span className="font-semibold">localhost:8000</span>
              </div>
            </div>
          </aside>

          <main className="flex-1">
            <div className="rounded-2xl border bg-white/80 p-5 shadow-sm backdrop-blur md:p-8">
              <Routes>
                <Route path="/" element={<Navigate to="/editor" replace />} />
                <Route path="/editor" element={<EditorPage />} />
                <Route path="/generate" element={<GeneratePage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  );
}

export default App;
