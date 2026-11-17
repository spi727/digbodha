const PATHS: Record<string, string> = {
  play: "M7 5.5v13a.6.6 0 0 0 .9.5l10.4-6.5a.6.6 0 0 0 0-1L7.9 5a.6.6 0 0 0-.9.5Z",
  pause: "M7.5 5h3v14h-3zM13.5 5h3v14h-3z",
  prev: "M6 6h2v12H6zM18 6.6v10.8a.6.6 0 0 1-.9.5L9.6 12.5a.6.6 0 0 1 0-1l7.5-5.4a.6.6 0 0 1 .9.5Z",
  next: "M16 6h2v12h-2zM6 6.6v10.8a.6.6 0 0 0 .9.5l7.5-5.4a.6.6 0 0 0 0-1L6.9 6.1a.6.6 0 0 0-.9.5Z",
  back: "M15 18l-6-6 6-6",
  arrow: "M5 12h14M13 6l6 6-6 6",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  close: "M6 6l12 12M18 6 6 18",
  alert: "M12 4 2.5 20h19L12 4ZM12 10v4M12 17.2v.1",
  skip: "M5 6l8 6-8 6zM15 6l8 6-8 6z",
  github: "M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.3-3.4-1.3-.5-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.4 1.1 2.9.8.1-.7.4-1.1.6-1.3-2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3 2.8 1a9.6 9.6 0 0 1 5 0c1.9-1.3 2.8-1 2.8-1 .5 1.4.2 2.4.1 2.7.6.7 1 1.6 1 2.7 0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.9V21c0 .3.2.6.7.5A10 10 0 0 0 12 2Z",
  check: "M5 12.5l4.5 4.5L19 7.5",
  layers: "M12 3 2 8l10 5 10-5-10-5ZM2 13l10 5 10-5",
  keyboard: "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10",
};

export function Icon({ name, size = 16, className = "" }: { name: keyof typeof PATHS | string; size?: number; className?: string }) {
  const filled = name === "play" || name === "pause" || name === "prev" || name === "next" || name === "skip" || name === "github";
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={`icon ${className}`}
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
