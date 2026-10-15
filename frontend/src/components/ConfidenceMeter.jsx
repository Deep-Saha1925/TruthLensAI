import { confidenceBarPercent, leadingSide } from '../utils/resultPresentation.js';

/**
 * Confidence = probability of the leading side. It is NOT a measure of correctness, so the
 * bar shows the margin over a coin flip and the caption says what the number does not mean.
 */
export default function ConfidenceMeter({ confidence, aiProbability, calibrated }) {
  const percent = Math.round(confidence * 100);
  const side = leadingSide(aiProbability).label;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold">Confidence in the leading side</h3>
        <p className="text-2xl font-bold tabular-nums">{percent}%</p>
      </div>

      <div
        role="meter"
        aria-label="Confidence in the leading side"
        aria-valuemin={50}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${percent}% toward ${side}`}
        className="mt-2 h-3 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-300"
          style={{ width: `${confidenceBarPercent(confidence)}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs text-muted" aria-hidden="true">
        <span>50% (coin flip)</span>
        <span>100%</span>
      </div>

      <p className="mt-2 text-sm text-muted">
        Leans toward <strong className="text-fg">{side}</strong>.{' '}
        {calibrated
          ? 'This number is not a guarantee that the lean is correct.'
          : 'This is a raw detector score, not a calibrated probability, and not a measure of how likely the lean is to be correct.'}
      </p>
    </div>
  );
}