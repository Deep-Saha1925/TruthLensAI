import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      <h1 className="text-2xl font-bold">Page not found</h1>
      <p className="mt-3 text-muted">That page does not exist.</p>
      <Link
        to="/"
        className="mt-6 inline-block rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg"
      >
        Go to the uploader
      </Link>
    </div>
  );
}