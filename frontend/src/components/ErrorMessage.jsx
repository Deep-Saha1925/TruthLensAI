import { AlertTriangle, X } from 'lucide-react';

export default function ErrorMessage({
  title = 'Something went wrong',
  message,
  requestId,
  onDismiss,
  onRetry,
}) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-xl border border-danger-border bg-danger-bg p-4 text-danger-fg"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{title}</p>
        <p className="text-sm">{message}</p>
        {requestId && (
          <p className="mt-1 break-all text-xs opacity-80">Reference: {requestId}</p>
        )}
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 rounded-lg border border-danger-border px-3 py-1.5 text-sm font-medium hover:bg-black/10"
          >
            Try again
          </button>
        )}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="rounded p-1 hover:bg-black/10"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}