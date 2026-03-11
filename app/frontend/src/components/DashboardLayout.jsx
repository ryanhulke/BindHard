import { useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";

const VIEWER_MODES = [
  { id: "molstar", label: "Mol*" },
  { id: "metrics", label: "Metrics" },
  { id: "video", label: "Video" },
];

export default function DashboardLayout() {
  const navigate = useNavigate();
  const [viewerMode, setViewerMode] = useState("molstar");
  const activeIndex = VIEWER_MODES.findIndex((mode) => mode.id === viewerMode);
  const translateClassByIndex = ["translate-x-0", "translate-x-[70px]", "translate-x-[140px]"];

  return (
    <div className="min-h-screen" style={{ background: "#080a0f" }}>
      {/* Top navigation bar */}
      <nav
        className="sticky top-0 z-50 border-b border-white/5 backdrop-blur-md"
        style={{ background: "rgba(8,10,15,0.9)" }}
      >
        <div className="flex items-center h-14 px-5 gap-4 max-w-[1400px] mx-auto">
          {/* Home / Logo */}
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 hover:opacity-70 transition-opacity mr-2"
          >
            <img src="/logo.png" alt="" className="w-5 h-5" />
            <span className="text-white font-bold text-sm tracking-wide">Bind Hard</span>
          </button>

          <div className="h-5 w-px bg-white/10" />

          {/* Right side mode toggle */}
          <div className="ml-auto flex items-center gap-3">
            <span className="text-[10px] text-white/30 tracking-widest uppercase">Viewer</span>
            <div
              role="tablist"
              aria-label="Viewer mode"
              className="relative h-7 w-[210px] rounded-full border border-white/15 bg-white/5"
            >
              <span
                className={`pointer-events-none absolute left-0.5 top-0.5 h-6 w-[68px] rounded-full bg-blue-500/90 transition-transform ${
                  translateClassByIndex[Math.max(0, activeIndex)] || "translate-x-0"
                }`}
              />
              <div className="relative z-10 grid h-full grid-cols-3">
                {VIEWER_MODES.map((mode) => {
                  const isActive = viewerMode === mode.id;
                  return (
                    <button
                      key={mode.id}
                      type="button"
                      role="tab"
                      aria-selected={isActive}
                      aria-label={`Switch viewer to ${mode.label}`}
                      onClick={() => setViewerMode(mode.id)}
                      className={`text-[10px] font-semibold uppercase tracking-widest transition-colors ${
                        isActive ? "text-white" : "text-white/55 hover:text-white/80"
                      }`}
                    >
                      {mode.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Page content */}
      <Outlet context={{ viewerMode, setViewerMode }} />
    </div>
  );
}
