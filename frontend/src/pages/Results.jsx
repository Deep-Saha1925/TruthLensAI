import { Link, useLocation, useParams } from 'react-router-dom';

import DisclaimerNotice from '../components/DisclaimerNotice.jsx';
import ErrorMessage from '../components/ErrorMessage.jsx';
import SkeletonCard from '../components/SkeletonCard.jsx';
import { useResult } from '../hooks/useResult.js';

const LinkButton = ({ to, children }) => (
  <Link
    to={to}
    className="inline-block rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity hover:opacity-90"
  >
    {children}
  </Link>
);

/** Phase 6: a plain summary that proves the connection. Phase 7 replaces it with the dashboard. */
export default function Results() {
  const { resultId } = useParams();
  const location = useLocation();
  const { status, result, error, retry } = useResult(resultId, location.state?.result ?? null);

  if (status === 'loading') {
    return (
      <div className="mx-auto max-w-3xl space-y-4 px-4 py-12" aria-busy="true">
        <p role="status" className="sr-only">
          Loading result
        </p>
        <SkeletonCard lines={2} />
        <SkeletonCard lines={4} />
      </div>
    );
  }

  if (status === 'error') {
    const expired = error.status === 404;
    return (
      <div className="mx-auto max-w-3xl space-y-6 px-4 py-12">
        <ErrorMessage
          title={expired ? 'Result not found or expired' : 'Could not load the result'}
          message={
            expired
              ? 'Results are kept in server memory for a short time (15 minutes by default) and are lost when the server restarts. Analyze the file again to get a new result.'
              : error.message
          }
          requestId={error.requestId}
          onRetry={expired ? undefined : retry}
        />
        <LinkButton to="/">Analyze a file</LinkButton>
      </div>
    );
  }

  const aiPercent = Math.round(result.ai_probability * 100);
  const realPercent = Math.round(result.real_probability * 100);

  return (
    <div className="mx-auto max-w-3xl space-y-6 px-4 py-12">
      <header>
        <p className="text-sm font-medium text-accent">
          {result.media_type === 'video' ? 'Video' : 'Image'} analysis
        </p>
        <h1 className="mt-1 text-3xl font-bold">{result.classification_label}</h1>
        <p className="mt-2 text-muted">
          Estimated probability of AI generation or manipulation: <strong>{aiPercent}%</strong> (real:{' '}
          {realPercent}%). Risk level: <strong>{result.risk_level}</strong>.
        </p>
      </header>

      <p
        role="note"
        className="rounded-xl border border-warn-border bg-warn-bg p-4 text-sm text-warn-fg"
      >
        {result.tier_notice}
      </p>

      <dl className="grid gap-3 rounded-2xl border bg-surface p-5 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-muted">Model</dt>
          <dd className="font-medium">
            {result.model} v{result.detector_version}
          </dd>
        </div>
        <div>
          <dt className="text-muted">Processing time</dt>
          <dd className="font-medium">{result.processing_time_ms} ms</dd>
        </div>
        {result.media_type === 'video' && (
          <div>
            <dt className="text-muted">Frames analyzed</dt>
            <dd className="font-medium">{result.frames_analyzed}</dd>
          </div>
        )}
        <div>
          <dt className="text-muted">Score is calibrated</dt>
          <dd className="font-medium">{result.score_is_calibrated ? 'Yes' : 'No'}</dd>
        </div>
      </dl>

      <p className="text-sm text-muted">
        The full results dashboard (charts, timeline, metadata, explainability) is built in Phase
        7. The raw response is below so you can check what the backend returned.
      </p>

      <details className="rounded-2xl border bg-surface p-4">
        <summary className="cursor-pointer font-medium">Raw response (JSON)</summary>
        <pre className="mt-3 max-h-96 overflow-auto rounded-lg bg-surface-2 p-3 text-xs">
          {JSON.stringify(result, null, 2)}
        </pre>
      </details>

      <DisclaimerNotice />
      <LinkButton to="/">Analyze another file</LinkButton>
    </div>
  );
}