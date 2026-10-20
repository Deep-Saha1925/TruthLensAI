import { useMemo } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';

import DetectionResult from '../components/DetectionResult.jsx';
import ErrorMessage from '../components/ErrorMessage.jsx';
import SkeletonCard from '../components/SkeletonCard.jsx';
import { useResult } from '../hooks/useResult.js';
import { useServerInfo } from '../hooks/useServerInfo.js';
import { recallMedia } from '../utils/mediaStore.js';

const LinkButton = ({ to, children }) => (
  <Link
    to={to}
    className="inline-block rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity hover:opacity-90"
  >
    {children}
  </Link>
);

export default function Results() {
  const { resultId } = useParams();
  const location = useLocation();
  const { status, result, error, retry } = useResult(resultId, location.state?.result ?? null);
  const serverInfo = useServerInfo(); // for the model's measured performance
  // The original file exists only in the browser session that uploaded it.
  const media = useMemo(() => recallMedia(resultId), [resultId]);

  if (status === 'loading') {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-4 py-12" aria-busy="true">
        <p role="status" className="sr-only">
          Loading result
        </p>
        <SkeletonCard lines={2} />
        <div className="grid gap-4 md:grid-cols-2">
          <SkeletonCard lines={5} />
          <SkeletonCard lines={5} />
        </div>
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

  return <DetectionResult result={result} media={media} detector={serverInfo.detector} />;
}