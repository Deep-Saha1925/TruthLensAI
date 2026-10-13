import { Info } from 'lucide-react';

export const DISCLAIMER_TEXT =
  'AI detection is probabilistic. Results should not be treated as definitive proof of authenticity or manipulation.';

export default function DisclaimerNotice({ className = '' }) {
  return (
    <p role="note" className={`flex items-start gap-2 text-sm text-muted ${className}`}>
      <Info className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
      <span>{DISCLAIMER_TEXT}</span>
    </p>
  );
}