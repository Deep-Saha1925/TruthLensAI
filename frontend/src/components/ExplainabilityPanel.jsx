import { Minus, TrendingDown, TrendingUp } from 'lucide-react';

import {
  DIRECTION_TEXT,
  formatSignalValue,
  splitSignals,
} from '../utils/resultPresentation.js';

const SOURCE_LABEL = {
  heuristic: 'Heuristic measurement',
  forensic: 'Forensic / metadata',
  model: 'Model output',
};

const DIRECTION_ICON = {
  toward_synthetic: TrendingUp,
  toward_real: TrendingDown,
  neutral: Minus,
};

const LEAD = {
  DEMO_HEURISTIC:
    'This prototype has no trained model. Its score comes from the simple measurements below, which are not evidence of what a trained detector would see.',
  PRETRAINED_ML:
    "The items below describe what contributed to a third-party model's score. They show where the score came from, not proof about the media.",
  TRAINED:
    "The items below describe what contributed to the model's score. They show where the score came from, not proof about the media.",
};

function SignalItem({ signal }) {
  const value = formatSignalValue(signal);
  const Icon = DIRECTION_ICON[signal.direction] ?? Minus;

  return (
    <li className="rounded-xl border p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h4 className="font-semibold">{signal.title}</h4>
        <span className="rounded-full bg-surface-2 px-2 py-0.5 text-xs text-muted">
          {SOURCE_LABEL[signal.source] ?? signal.source}
        </span>
      </div>
      {value && <p className="mt-1 text-sm font-medium tabular-nums">{value}</p>}
      {signal.scored && (
        <p className="mt-1 flex items-center gap-1.5 text-sm">
          <Icon className="h-4 w-4" aria-hidden="true" />
          {DIRECTION_TEXT[signal.direction] ?? DIRECTION_TEXT.neutral}
        </p>
      )}
      <p className="mt-2 text-sm text-muted">{signal.detail}</p>
    </li>
  );
}

/** Phase 8 will pass a real heatmap here for detectors that support one. Nothing is faked. */
export default function ExplainabilityPanel({ explanation, notice, tier, mediaType }) {
  const { scored, context } = splitSignals(explanation);

  return (
    <section aria-labelledby="explain-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="explain-title" className="text-lg font-bold">
        Why did the model predict this?
      </h2>
      <p className="mt-1 text-sm text-muted">{LEAD[tier] ?? LEAD.TRAINED}</p>
      {notice && <p className="mt-2 text-sm text-muted">{notice}</p>}

      <h3 className="mt-5 text-sm font-semibold">Items that fed the score</h3>
      {scored.length > 0 ? (
        <ul className="mt-2 space-y-3">
          {scored.map((signal) => (
            <SignalItem key={signal.key} signal={signal} />
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-sm text-muted">No item contributed to this score.</p>
      )}

      {context.length > 0 && (
        <>
          <h3 className="mt-5 text-sm font-semibold">Context only (not used in the score)</h3>
          <ul className="mt-2 space-y-3">
            {context.map((signal) => (
              <SignalItem key={signal.key} signal={signal} />
            ))}
          </ul>
        </>
      )}

      <h3 className="mt-5 text-sm font-semibold">Heatmap</h3>
      <p className="mt-2 rounded-xl border border-dashed p-4 text-sm text-muted">
        No heatmap is available for this result. Heatmaps are shown only when the active detector
        can produce a reliable one, and a heatmap would show where a model looked, not proof of
        manipulation.
        {mediaType === 'video' && ' For videos, see the highest-scoring frames above.'}
      </p>
    </section>
  );
}