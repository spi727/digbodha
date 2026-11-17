import type { Risk } from "./data";

export const RISK_COLOR: Record<Risk, string> = {
  safe: "var(--safe)",
  uncertain: "var(--uncertain)",
  caution: "var(--caution)",
  danger: "var(--danger)",
};

// Canvas can't read CSS variables cheaply per frame; keep raw values in sync with styles.css.
export const RISK_HEX: Record<Risk, string> = {
  safe: "#3ddc97",
  uncertain: "#b18cff",
  caution: "#ffb547",
  danger: "#ff5a5f",
};

export const RISK_COPY: Record<Risk, { name: string; rule: string }> = {
  danger: { name: "Danger", rule: "On a collision course and contact in under 2 s." },
  caution: { name: "Caution", rule: "On course with contact under 4 s, or a close pass under 1.5 s." },
  uncertain: { name: "Uncertain", rule: "Looks like caution or danger, but a reliability check failed." },
  safe: { name: "Safe", rule: "No caution or danger condition applies." },
};

export const REASON_COPY: Record<string, string> = {
  warming_up: "Fewer than 5 observations, so no estimate yet",
  too_small: "Too small in the image to measure growth reliably",
  noisy_expansion: "Size measurements are too noisy for a stable fit",
  size_jump: "Sudden size jump, likely a segmentation glitch",
  inconsistent_growth: "Height and width disagree on growth",
  camera_unreliable: "Camera-motion estimate is unreliable",
  cut_top_bottom: "Cut off at the frame's top or bottom edge",
  cut_and_turning: "Cut off at the frame edge while turning",
};

export const MOTION_COPY: Record<string, string> = {
  still: "Still",
  moving_left: "Moving left",
  moving_right: "Moving right",
  moving: "Moving",
  unknown: "Measuring",
};

export const fmtS = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "–" : `${v.toFixed(d)}s`);
export const fmtNum = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "–" : v.toFixed(d));
export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
};
export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
