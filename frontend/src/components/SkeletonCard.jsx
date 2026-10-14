/** Pulsing placeholder shown while content is loading. Hidden from screen readers. */
export default function SkeletonCard({ lines = 3, className = '' }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded-2xl border bg-surface p-5 ${className}`}
    >
      <div className="h-5 w-1/3 rounded bg-surface-2" />
      <div className="mt-4 space-y-2">
        {Array.from({ length: lines }, (_, index) => (
          <div
            key={index}
            className="h-3 rounded bg-surface-2"
            style={{ width: `${Math.max(30, 92 - index * 14)}%` }}
          />
        ))}
      </div>
    </div>
  );
}