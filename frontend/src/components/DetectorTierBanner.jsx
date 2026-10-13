import { AlertTriangle, FlaskConical, ServerOff, ShieldCheck } from 'lucide-react';

const TONES = {
  warn: 'border-warn-border bg-warn-bg text-warn-fg',
  danger: 'border-danger-border bg-danger-bg text-danger-fg',
  info: 'border-info-border bg-info-bg text-info-fg',
};

const TIER_PRESENTATION = {
  DEMO_HEURISTIC: { label: 'Prototype mode', tone: 'warn', Icon: FlaskConical },
  PRETRAINED_ML: { label: 'Third-party model', tone: 'info', Icon: ShieldCheck },
  TRAINED: { label: 'Trained model', tone: 'info', Icon: ShieldCheck },
};

function Banner({ tone, Icon, label, children, alert = false }) {
  return (
    <div
      role={alert ? 'alert' : 'status'}
      className={`border-b px-4 py-2 text-sm ${TONES[tone]}`}
    >
      <p className="mx-auto flex max-w-5xl items-start gap-2">
        <Icon className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
        <span>
          <strong>{label}:</strong> {children}
        </span>
      </p>
    </div>
  );
}

/** Always tells the user what kind of detector is behind the results. */
export default function DetectorTierBanner({ status, detector }) {
  if (status === 'loading') {
    return <div aria-hidden="true" className="h-9 animate-pulse border-b bg-surface-2" />;
  }

  if (status === 'unreachable') {
    return (
      <Banner tone="danger" Icon={ServerOff} label="Server unreachable" alert>
        Cannot reach the analysis server. Start the backend (<code>python -m app</code>) and
        reload this page.
      </Banner>
    );
  }

  if (!detector) {
    return (
      <Banner tone="danger" Icon={AlertTriangle} label="Model not loaded" alert>
        The server is running but its analysis model failed to load. Uploads cannot be analyzed.
        Check the server logs.
      </Banner>
    );
  }

  const { label, tone, Icon } = TIER_PRESENTATION[detector.tier] ?? TIER_PRESENTATION.DEMO_HEURISTIC;
  return (
    <Banner tone={tone} Icon={Icon} label={label}>
      {detector.tier_notice}
    </Banner>
  );
}