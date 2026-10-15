import { Clock } from 'lucide-react';

import { formatPercent, formatTimestamp } from '../utils/formatting.js';
import { TONE_CLASSES } from '../utils/resultPresentation.js';

export default function SuspiciousFrames({ frames, threshold, canSeek, onSeek }) {
  const flaggedCount = frames.filter((frame) => frame.flagged).length;

  return (
    <section aria-labelledby="top-frames-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="top-frames-title" className="text-lg font-bold">
        Highest-scoring frames
      </h2>
      <p className="mt-1 text-sm text-muted">
        {flaggedCount > 0
          ? `${flaggedCount} of these frames scored at or above the ${formatPercent(threshold)} synthetic threshold. `
          : `No analyzed frame reached the ${formatPercent(threshold)} synthetic threshold. `}
        A high score means the detector leaned toward "synthetic" for that frame; it is not proof
        that the frame was manipulated.
        {!canSeek &&
          ' Jumping to a frame needs the original video, which is only available in the browser session that uploaded it and only for formats your browser can play.'}
      </p>

      <ol className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {frames.map((frame, index) => (
          <li key={frame.frame_number} className="flex flex-col rounded-xl border p-4">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-semibold">
                #{index + 1} · Frame {frame.frame_number}
              </p>
              <span
                className={`shrink-0 rounded-full border px-2 py-0.5 text-xs font-medium ${
                  frame.flagged ? TONE_CLASSES.danger : 'bg-surface-2 text-muted'
                }`}
              >
                {frame.flagged ? 'At/above threshold' : 'Below threshold'}
              </span>
            </div>

            <p className="mt-2 flex items-center gap-1 text-sm text-muted">
              <Clock className="h-4 w-4" aria-hidden="true" />
              {formatTimestamp(frame.timestamp_sec)}
            </p>

            <div className="mt-3">
              <div className="flex items-baseline justify-between text-sm">
                <span className="text-muted">AI score</span>
                <span className="font-bold tabular-nums">{formatPercent(frame.ai_probability)}</span>
              </div>
              <div
                role="meter"
                aria-label={`AI score for frame ${frame.frame_number}`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(frame.ai_probability * 100)}
                className="mt-1 h-2 overflow-hidden rounded-full bg-surface-2"
              >
                <div
                  className="h-full rounded-full bg-flag"
                  style={{ width: `${Math.round(frame.ai_probability * 100)}%` }}
                />
              </div>
            </div>

            <button
              type="button"
              disabled={!canSeek}
              onClick={() => onSeek(frame.timestamp_sec)}
              className="mt-4 rounded-lg border px-3 py-2 text-sm font-medium transition-colors hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Jump to this moment
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}