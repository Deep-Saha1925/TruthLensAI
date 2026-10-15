import {
  SYNTHETIC_LABEL,
  ZONE_LABEL,
  scoreZone,
  splitPercent,
} from '../utils/resultPresentation.js';

// A diagonal hatch makes the "AI" bar distinguishable from the "Real" bar without colour.
const HATCH =
  'repeating-linear-gradient(45deg, rgba(255,255,255,0.4) 0 4px, transparent 4px 8px)';

const ZONE_CLASS = {
  real: 'border-ok-border bg-ok-bg',
  uncertain: 'border-warn-border bg-warn-bg',
  synthetic: 'border-danger-border bg-danger-bg',
};

function Bar({ label, percent, fillClass, hatched = false }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-lg font-bold tabular-nums">{percent}%</span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        className="mt-1 h-4 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${fillClass}`}
          style={{ width: `${percent}%`, backgroundImage: hatched ? HATCH : undefined }}
        />
      </div>
    </div>
  );
}

export default function ProbabilityChart({ aiProbability, uncertainBand, calibrated }) {
  const [low, high] = uncertainBand;
  const { ai, real } = splitPercent(aiProbability);
  const zone = scoreZone(aiProbability, uncertainBand);

  const lowPct = low * 100;
  const highPct = high * 100;
  const markerLeft = Math.min(100, Math.max(0, aiProbability * 100));
  const labelLeft = Math.min(92, Math.max(8, markerLeft)); // keep the label inside the card

  return (
    <section aria-labelledby="probability-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="probability-title" className="text-lg font-bold">
        Real vs. AI score
      </h2>
      <p className="mt-1 text-sm text-muted">
        {calibrated
          ? 'Estimated probabilities from a calibrated detector.'
          : 'Raw detector scores. They add up to 100% but are not calibrated probabilities.'}
      </p>

      <div className="mt-4 space-y-4">
        <Bar label={SYNTHETIC_LABEL} percent={ai} fillClass="bg-flag" hatched />
        <Bar label="Real" percent={real} fillClass="bg-real" />
      </div>

      <h3 className="mt-6 text-sm font-semibold">Where the score falls</h3>
      <div className="relative mt-8" aria-hidden="true">
        <div
          className="absolute -top-6 -translate-x-1/2 whitespace-nowrap text-xs font-semibold"
          style={{ left: `${labelLeft}%` }}
        >
          Score {ai}%
        </div>
        <div className="flex h-8 overflow-hidden rounded-lg border">
          <div className={`border-r ${ZONE_CLASS.real}`} style={{ width: `${lowPct}%` }} />
          <div
            className={`border-r ${ZONE_CLASS.uncertain}`}
            style={{ width: `${highPct - lowPct}%` }}
          />
          <div className={ZONE_CLASS.synthetic} style={{ width: `${100 - highPct}%` }} />
        </div>
        <div
          className="absolute w-1 -translate-x-1/2 rounded bg-fg"
          style={{ left: `${markerLeft}%`, top: '-4px', bottom: '-4px' }}
        />
      </div>

      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
        <li>
          <strong className="text-fg">{ZONE_LABEL.real}:</strong> 0–{Math.round(lowPct)}%
        </li>
        <li>
          <strong className="text-fg">{ZONE_LABEL.uncertain}:</strong> {Math.round(lowPct)}–
          {Math.round(highPct)}%
        </li>
        <li>
          <strong className="text-fg">{ZONE_LABEL.synthetic}:</strong> {Math.round(highPct)}–100%
        </li>
      </ul>
      <p className="mt-2 text-sm">
        A score of {ai}% falls in the <strong>{ZONE_LABEL[zone]}</strong> zone. Scores near the
        middle are reported as uncertain instead of forcing a side.
      </p>
    </section>
  );
}