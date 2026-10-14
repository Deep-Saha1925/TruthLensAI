/** Pure presentation helpers for the results dashboard (no React, easy to test). */

export const SYNTHETIC_LABEL = 'AI-generated or manipulated';

export const TONE_CLASSES = {
  ok: 'border-ok-border bg-ok-bg text-ok-fg',
  warn: 'border-warn-border bg-warn-bg text-warn-fg',
  danger: 'border-danger-border bg-danger-bg text-danger-fg',
  info: 'border-info-border bg-info-bg text-info-fg',
};

export const CLASSIFICATION_TONE = {
  LIKELY_REAL: 'ok',
  UNCERTAIN: 'warn',
  LIKELY_SYNTHETIC: 'danger',
};

export const RISK_TONE = { LOW: 'ok', MEDIUM: 'warn', HIGH: 'danger' };
export const RISK_LABEL = { LOW: 'Low risk', MEDIUM: 'Medium risk', HIGH: 'High risk' };

export const TIER_TONE = { DEMO_HEURISTIC: 'warn', PRETRAINED_ML: 'info', TRAINED: 'info' };
export const TIER_LABEL = {
  DEMO_HEURISTIC: 'Prototype mode',
  PRETRAINED_ML: 'Third-party model',
  TRAINED: 'Trained model',
};

export const ZONE_LABEL = {
  real: 'Likely real',
  uncertain: 'Uncertain',
  synthetic: 'Likely synthetic',
};

export const DIRECTION_TEXT = {
  toward_synthetic: 'Leans toward synthetic',
  toward_real: 'Leans toward real',
  neutral: 'No clear lean',
};

/** Whole percentages that always add up to 100 (rounding both sides can give 101). */
export function splitPercent(aiProbability) {
  const clamped = Math.min(1, Math.max(0, Number(aiProbability) || 0));
  const ai = Math.round(clamped * 100);
  return { ai, real: 100 - ai };
}

export function leadingSide(aiProbability) {
  return aiProbability >= 0.5
    ? { key: 'synthetic', label: SYNTHETIC_LABEL }
    : { key: 'real', label: 'Real' };
}

/** Same boundaries as the backend's classify(): the band edges themselves are not "uncertain". */
export function scoreZone(aiProbability, [low, high]) {
  if (aiProbability >= high) return 'synthetic';
  if (aiProbability <= low) return 'real';
  return 'uncertain';
}

/** Confidence is always >= 0.5; the meter shows the margin over a coin flip (0.5 -> 0, 1 -> 100). */
export function confidenceBarPercent(confidence) {
  return Math.min(100, Math.max(0, (confidence - 0.5) * 200));
}

export function buildTimelineData(frameResults) {
  return frameResults.map((frame) => ({
    t: frame.timestamp_sec,
    p: frame.ai_probability,
    frame: frame.frame_number,
    flagged: frame.flagged,
    selection: frame.selection,
  }));
}

/** Right edge of the timeline's time axis (never zero, so the axis always has width). */
export function timelineMaxTime(points, durationSec) {
  const last = points.length > 0 ? Math.max(...points.map((point) => point.t)) : 0;
  const duration = Number.isFinite(durationSec) ? durationSec : 0;
  return Math.max(last, duration, 1);
}

export function splitSignals(explanation = []) {
  return {
    scored: explanation.filter((signal) => signal.scored),
    context: explanation.filter((signal) => !signal.scored),
  };
}

export function formatSignalValue(signal) {
  if (signal.value == null) return null;
  return `${signal.value}${signal.unit ? ` ${signal.unit}` : ''}`;
}