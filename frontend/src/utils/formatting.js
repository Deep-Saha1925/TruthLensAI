export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

/** 65.4 -> "1:05" */
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = String(whole % 60).padStart(2, '0');
  return `${minutes}:${rest}`;
}

/** 0.4188 -> "42%". Input is a probability in [0, 1]; out-of-range values are clamped. */
export function formatPercent(probability, digits = 0) {
  if (!Number.isFinite(probability)) return '—';
  const clamped = Math.min(1, Math.max(0, probability));
  return `${(clamped * 100).toFixed(digits)}%`;
}

/** Position in a video with tenths of a second: 4 -> "0:04.0", 65.4 -> "1:05.4" */
export function formatTimestamp(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const tenths = Math.round(seconds * 10);
  const minutes = Math.floor(tenths / 600);
  const rest = ((tenths % 600) / 10).toFixed(1).padStart(4, '0');
  return `${minutes}:${rest}`;
}

export function formatDateTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}