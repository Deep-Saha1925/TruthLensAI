import { describe, expect, it } from 'vitest';

import {
  evaluationApplies,
  formatMetric,
  parseEvaluation,
} from '../utils/modelEvaluation.js';

const EVALUATION = {
  test_images: 480,
  accuracy: 0.912,
  roc_auc: 0.9731,
  ece: 0.041,
  robust_auc_jpeg_q70: 0.88,
  robust_acc_jpeg_q70: 0.81,
  robust_auc_downscale_50: 0.84,
  dataset: 'My dataset',
  warning: 'Evaluated on only 480 test images.',
  custom_note: 'hello',
};

describe('formatMetric', () => {
  it('formats by kind and guards against bad values', () => {
    expect(formatMetric(0.912, 'percent')).toBe('91.2%');
    expect(formatMetric(0.9731, 'number')).toBe('0.973');
    expect(formatMetric(479.6, 'count')).toBe('480');
    expect(formatMetric(undefined, 'percent')).toBe('—');
    expect(formatMetric(Number.NaN, 'number')).toBe('—');
  });
});

describe('parseEvaluation', () => {
  it('returns null when there is nothing to show', () => {
    expect(parseEvaluation(null)).toBeNull();
    expect(parseEvaluation({})).toBeNull();
  });

  it('collects headline metrics, robustness rows, notes and the warning', () => {
    const parsed = parseEvaluation(EVALUATION);
    expect(parsed.headline.map((m) => m.key)).toEqual(['accuracy', 'roc_auc', 'ece', 'test_images']);
    expect(parsed.headline[0].text).toBe('91.2%');
    expect(parsed.warning).toBe('Evaluated on only 480 test images.');
    expect(parsed.notes.map((n) => n.key)).toEqual(['dataset']);
    expect(parsed.other).toEqual([{ key: 'custom_note', label: 'custom_note', text: 'hello' }]);
    expect(parsed.baseline).toEqual({ auc: '0.973', accuracy: '91.2%' });
  });

  it('shows a missing half of a robustness row as a dash', () => {
    const rows = parseEvaluation(EVALUATION).robustness;
    expect(rows.map((r) => r.key)).toEqual(['jpeg_q70', 'downscale_50']);
    expect(rows[0]).toMatchObject({ auc: '0.880', accuracy: '81.0%' });
    expect(rows[1]).toMatchObject({ auc: '0.840', accuracy: '—' });
  });
});

describe('evaluationApplies', () => {
  const detector = { name: 'm', version: '1', evaluation: EVALUATION };
  const result = { model: 'm', detector_version: '1' };

  it('requires the same model and version as the result', () => {
    expect(evaluationApplies(detector, result)).toBe(true);
    expect(evaluationApplies(detector, { ...result, detector_version: '2' })).toBe(false);
    expect(evaluationApplies(detector, { ...result, model: 'other' })).toBe(false);
  });

  it('is false without a detector or an evaluation', () => {
    expect(evaluationApplies(null, result)).toBe(false);
    expect(evaluationApplies({ ...detector, evaluation: null }, result)).toBe(false);
  });
});