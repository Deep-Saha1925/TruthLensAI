import { Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import SkeletonCard from './SkeletonCard.jsx';

/**
 * phase: 'uploading' (progress 0..1, or null if the browser cannot report it) | 'analyzing'.
 * Indeterminate while analyzing: the server gives no progress for a single analysis.
 */
export default function LoadingState({ phase, progress, kind, fileName, onCancel }) {
  const headingRef = useRef(null);
  const [elapsed, setElapsed] = useState(0);

  // The button that started the analysis just unmounted: move focus somewhere meaningful.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => setElapsed((seconds) => seconds + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const uploading = phase === 'uploading';
  const percent = uploading && progress != null ? Math.round(progress * 100) : null;
  const title = uploading ? 'Uploading…' : 'Analyzing…';
  const detail = uploading
    ? percent != null
      ? `${percent}% uploaded`
      : 'Sending the file to the server'
    : kind === 'video'
      ? 'Sampling frames and scoring each one. Videos can take a minute or more.'
      : 'Scoring the image. This usually takes a few seconds.';

  return (
    <div className="space-y-4 rounded-2xl border bg-surface p-6">
      <div className="flex items-start gap-3">
        <Loader2 className="mt-1 h-6 w-6 shrink-0 animate-spin text-accent" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          {/* Live region: changes only when the phase changes, not every second. */}
          <h3
            ref={headingRef}
            tabIndex={-1}
            role="status"
            className="text-lg font-semibold outline-none"
          >
            {title}
          </h3>
          <p className="text-sm text-muted">{detail}</p>
          {fileName && <p className="mt-1 break-all text-xs text-muted">{fileName}</p>}
        </div>
        <p className="shrink-0 text-sm tabular-nums text-muted" aria-hidden="true">
          {elapsed} s
        </p>
      </div>

      <div
        role="progressbar"
        aria-label={uploading ? 'Upload progress' : 'Analysis in progress'}
        aria-valuemin={percent != null ? 0 : undefined}
        aria-valuemax={percent != null ? 100 : undefined}
        aria-valuenow={percent ?? undefined}
        className="h-2 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className={`h-full rounded-full bg-accent transition-[width] duration-200 ${
            percent == null ? 'w-full animate-pulse' : ''
          }`}
          style={percent != null ? { width: `${percent}%` } : undefined}
        />
      </div>

      <div className="grid gap-3 sm:grid-cols-2" aria-hidden="true">
        <SkeletonCard lines={2} />
        <SkeletonCard lines={2} />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted">
          Cancelling stops waiting here; the server may still finish the analysis.
        </p>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-surface-2"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}