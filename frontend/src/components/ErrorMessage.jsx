import { AlertTriangle, X } from 'lucide-react';

export default function ErrorMessage({ title = 'Something went wrong', message, onDismiss }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-3 rounded-xl border border-danger-border bg-danger-bg p-4 text-danger-fg"
    >
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="font-semibold">{title}</p>
        <p className="text-sm">{message}</p>
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