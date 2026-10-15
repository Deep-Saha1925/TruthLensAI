import { Minus, TrendingDown, TrendingUp } from 'lucide-react';

import {
  DIRECTION_TEXT,
  formatSignalValue,
  splitSignals,
} from '../utils/resultPresentation.js';
import HeatmapOverlay from './HeatmapOverlay.jsx';

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

const NO_HEATMAP_FALLBACK = 'No heatmap is available for this result.';

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

/**
 * The heatmap appears only when the server returned one. Otherwise the server's own notice
 * explains why not (unsupported detector, disabled, or failed). Nothing is drawn locally.
 */
export default function ExplainabilityPanel({
  explanation,
  notice,
  tier,
  mediaType,
  heatmap,
  heatmapNotice,
  originalFile,
}) {
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

      <h3 className="mt-5 text-sm font-semibold">Heatmap (where the model looked)</h3>
      {heatmap && (
        <div className="mt-2">
          <HeatmapOverlay heatmap={heatmap} file={originalFile} />
        </div>
      )}
      <p
        className={
          heatmap
            ? 'mt-3 text-sm text-muted'
            : 'mt-2 rounded-xl border border-dashed p-4 text-sm text-muted'
        }
      >
        {heatmapNotice ?? NO_HEATMAP_FALLBACK}
        {!heatmap && mediaType === 'video' &&
          ' Heatmaps are computed for images only; for videos, see the highest-scoring frames above.'}
      </p>
    </section>
  );
}