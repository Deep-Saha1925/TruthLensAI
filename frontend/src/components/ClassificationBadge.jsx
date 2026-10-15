import { AlertTriangle, CheckCircle2, HelpCircle } from 'lucide-react';

import { CLASSIFICATION_TONE, TONE_CLASSES } from '../utils/resultPresentation.js';

const ICONS = { ok: CheckCircle2, warn: HelpCircle, danger: AlertTriangle };

/** Colour + icon + text: the meaning never depends on colour alone. */
export default function ClassificationBadge({ classification, label, large = false }) {
  const tone = CLASSIFICATION_TONE[classification] ?? 'warn';
  const Icon = ICONS[tone];

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border font-semibold ${TONE_CLASSES[tone]} ${
        large ? 'px-4 py-2 text-lg' : 'px-3 py-1 text-sm'
      }`}
    >
      <Icon className={large ? 'h-6 w-6' : 'h-4 w-4'} aria-hidden="true" />
      {label}
    </span>
  );
}