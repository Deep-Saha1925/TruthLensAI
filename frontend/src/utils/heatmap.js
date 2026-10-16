/** Same colours as the backend's colour stops (explainability.py). */
export const HEATMAP_LEGEND_GRADIENT =
  'linear-gradient(to right, #313695, #4575b4, #fee090, #f46d43, #a50026)';

export const DEFAULT_HEATMAP_OPACITY_PERCENT = 60;

/** Slider percent (0-100) -> CSS opacity (0-1). Invalid input falls back to the default. */
export function percentToOpacity(percent) {
  const value = Number(percent);
  if (!Number.isFinite(value)) return DEFAULT_HEATMAP_OPACITY_PERCENT / 100;
  return Math.min(1, Math.max(0, value / 100));
}

export function heatmapAspectRatio({ width, height }) {
  return `${width} / ${height}`;
}