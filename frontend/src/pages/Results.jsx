import { Link, useParams } from 'react-router-dom';

export default function Results() {
  const { resultId } = useParams();

  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="text-2xl font-bold">Results</h1>
      <p className="mt-3 text-muted">
        The results dashboard is built in Phase 7. Requested result:{' '}
        <code className="break-all">{resultId}</code>
      </p>
      <Link
        to="/"
        className="mt-6 inline-block rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg"
      >
        Back to upload
      </Link>
    </div>
  );
}