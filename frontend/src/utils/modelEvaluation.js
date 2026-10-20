/** Turns the flat `evaluation` dict from /api/models into something the UI can show. */

export const HEADLINE_METRICS = [
  ['accuracy', 'Accuracy', 'percent'],
  ['precision', 'Precision (of images called fake)', 'percent'],
  ['recall', 'Recall (of fake images found)', 'percent'],
  ['f1', 'F1 score', 'percent'],
  ['roc_auc', 'ROC-AUC', 'number'],
  ['ece', 'Calibration error (ECE, lower is better)', 'number'],
  ['uncertain_fraction', 'Share scored "uncertain"', 'percent'],
  ['accuracy_outside_uncertain_band', 'Accuracy on images not scored "uncertain"', 'percent'],
  ['test_images', 'Test images', 'count'],
];

export const ROBUSTNESS_LABELS = {
  jpeg_q90: 'JPEG, quality 90',
  jpeg_q70: 'JPEG, quality 70',
  jpeg_q50: 'JPEG, quality 50',
  downscale_50: 'Downscaled to 50%',
  downscale_25: 'Downscaled to 25%',
  blur_1_5: 'Blurred (sigma 1.5)',
};

const NOTE_KEYS = [
  ['dataset', 'Data'],
  ['trained_with', 'Trained with'],
  ['threshold', 'Cut-offs'],
  ['evaluated_at', 'Evaluated at'],
  ['manifest_sha256', 'Data manifest (SHA-256 prefix)'],
];

export function formatMetric(value, kind) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  if (kind === 'percent') return `${(value * 100).toFixed(1)}%`;
  if (kind === 'count') return String(Math.round(value));
  return value.toFixed(3);
}

const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);

/** Returns null when there is nothing worth showing. */
export function parseEvaluation(evaluation) {
  if (!evaluation || typeof evaluation !== 'object') return null;

  const headline = HEADLINE_METRICS.filter(([key]) => isNumber(evaluation[key])).map(
    ([key, label, kind]) => ({ key, label, text: formatMetric(evaluation[key], kind) }),
  );

  const robustness = Object.entries(ROBUSTNESS_LABELS)
    .filter(
      ([key]) => isNumber(evaluation[`robust_auc_${key}`]) || isNumber(evaluation[`robust_acc_${key}`]),
    )
    .map(([key, label]) => ({
      key,
      label,
      auc: formatMetric(evaluation[`robust_auc_${key}`], 'number'),
      accuracy: formatMetric(evaluation[`robust_acc_${key}`], 'percent'),
    }));

  const notes = NOTE_KEYS.filter(([key]) => typeof evaluation[key] === 'string' && evaluation[key]).map(
    ([key, label]) => ({ key, label, text: evaluation[key] }),
  );

  const known = new Set([
    'warning',
    ...HEADLINE_METRICS.map(([key]) => key),
    ...NOTE_KEYS.map(([key]) => key),
    ...Object.keys(ROBUSTNESS_LABELS).flatMap((key) => [`robust_auc_${key}`, `robust_acc_${key}`]),
  ]);
  const other = Object.entries(evaluation)
    .filter(([key]) => !known.has(key))
    .map(([key, value]) => ({ key, label: key, text: String(value) }));

  const warning = typeof evaluation.warning === 'string' && evaluation.warning ? evaluation.warning : null;
  if (!headline.length && !robustness.length && !notes.length && !other.length && !warning) return null;

  return {
    headline,
    robustness,
    notes,
    other,
    warning,
    baseline: {
      auc: formatMetric(evaluation.roc_auc, 'number'),
      accuracy: formatMetric(evaluation.accuracy, 'percent'),
    },
  };
}

/** Only show a model's metrics next to a result produced by that same model and version. */
export function evaluationApplies(detector, result) {
  return Boolean(
    detector &&
      detector.evaluation &&
      result &&
      detector.name === result.model &&
      detector.version === result.detector_version,
  );
}