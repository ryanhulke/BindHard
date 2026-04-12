import { useEffect, useRef, useState } from "react";

export default function PageLoadProgress() {
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(true);
  const progressRef = useRef(0);

  useEffect(() => {
    let frameId = 0;
    let timeoutId = 0;
    let startTime = 0;
    let cleanupLoadListener = null;

    const animateTo = (target, duration, onComplete) => {
      const initial = progressRef.current;

      const step = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const elapsed = timestamp - startTime;
        const next = Math.min(1, elapsed / duration);
        const eased = 1 - Math.pow(1 - next, 3);
        const value = initial + (target - initial) * eased;
        progressRef.current = value;
        setProgress(value);

        if (next < 1) {
          frameId = window.requestAnimationFrame(step);
          return;
        }

        startTime = 0;
        onComplete?.();
      };

      frameId = window.requestAnimationFrame(step);
    };

    animateTo(88, 900, () => {
      const finish = () =>
        animateTo(100, 420, () => {
          timeoutId = window.setTimeout(() => setVisible(false), 220);
        });

      if (document.readyState === "complete") {
        finish();
        return;
      }

      window.addEventListener("load", finish, { once: true });
      cleanupLoadListener = () => window.removeEventListener("load", finish);
    });

    return () => {
      window.cancelAnimationFrame(frameId);
      window.clearTimeout(timeoutId);
      cleanupLoadListener?.();
    };
  }, []);

  if (!visible) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-[80] h-1 bg-transparent">
      <div
        className="h-full bg-teal-400 shadow-[0_0_12px_rgba(45,212,191,0.7)] transition-opacity duration-200"
        style={{
          width: `${progress}%`,
          opacity: progress >= 100 ? 0 : 1,
        }}
      />
    </div>
  );
}
