import { AlertTriangle, ShieldAlert, ShieldCheck } from 'lucide-react';

import { RISK_LABEL, RISK_TONE, TONE_CLASSES } from '../utils/resultPresentation.js';

const ICONS = { ok: ShieldCheck, warn: ShieldAlert, danger: AlertTriangle };

export default function RiskBadge({ level }) {
  const tone = RISK_TONE[level] ?? 'warn';
  const Icon = ICONS[tone];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-semibold ${TONE_CLASSES[tone]}`}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      {RISK_LABEL[level] ?? level}
    </span>
  );
}