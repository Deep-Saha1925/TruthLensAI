import { beforeEach, describe, expect, it } from 'vitest';

import { formatDateTime, formatPercent, formatTimestamp } from '../utils/formatting.js';
import { forgetMedia, recallMedia, rememberMedia } from '../utils/mediaStore.js';
import {
  buildTimelineData,
  confidenceBarPercent,
  formatSignalValue,
  leadingSide,
  scoreZone,
  splitPercent,
  splitSignals,
  timelineMaxTime,
} from '../utils/resultPresentation.js';

describe('formatting used by the dashboard', () => {
  it('formats probabilities as percentages and clamps out-of-range input', () => {
    expect(formatPercent(0.4188)).toBe('42%');
    expect(formatPercent(0.4188, 1)).toBe('41.9%');
    expect(formatPercent(1.5)).toBe('100%');
    expect(formatPercent(-1)).toBe('0%');
    expect(formatPercent(Number.NaN)).toBe('—');
  });

  it('formats video timestamps with tenths of a second', () => {
    expect(formatTimestamp(4)).toBe('0:04.0');
    expect(formatTimestamp(65.4)).toBe('1:05.4');
    expect(formatTimestamp(59.96)).toBe('1:00.0');
    expect(formatTimestamp(-1)).toBe('—');
  });

  it('survives an invalid date', () => {
    expect(formatDateTime('not a date')).toBe('—');
  });
});

describe('splitPercent', () => {
  it('always adds up to 100', () => {
    for (const p of [0, 0.01, 0.425, 0.4188, 0.5, 0.575, 0.999, 1]) {
      const { ai, real } = splitPercent(p);
      expect(ai + real).toBe(100);
    }
    expect(splitPercent(0.4188)).toEqual({ ai: 42, real: 58 });
  });

  it('treats invalid input as zero', () => {
    expect(splitPercent(Number.NaN)).toEqual({ ai: 0, real: 100 });
    expect(splitPercent(2)).toEqual({ ai: 100, real: 0 });
  });
});

describe('leadingSide and scoreZone', () => {
  it('names the leading side', () => {
    expect(leadingSide(0.5).key).toBe('synthetic');
    expect(leadingSide(0.49).key).toBe('real');
  });

  it("uses the backend's classification boundaries", () => {
    const band = [0.35, 0.65];
    expect(scoreZone(0.1, band)).toBe('real');
    expect(scoreZone(0.35, band)).toBe('real');
    expect(scoreZone(0.5, band)).toBe('uncertain');
    expect(scoreZone(0.65, band)).toBe('synthetic');
    expect(scoreZone(0.9, band)).toBe('synthetic');
  });
});

describe('confidenceBarPercent', () => {
  it('shows the margin over a coin flip', () => {
    expect(confidenceBarPercent(0.5)).toBe(0);
    expect(confidenceBarPercent(0.75)).toBeCloseTo(50);
    expect(confidenceBarPercent(1)).toBe(100);
    expect(confidenceBarPercent(0.4)).toBe(0);
  });
});

describe('timeline helpers', () => {
  const frames = [
    { timestamp_sec: 0.4, ai_probability: 0.3, frame_number: 4, flagged: false, selection: 'uniform' },
    { timestamp_sec: 4, ai_probability: 0.7, frame_number: 40, flagged: true, selection: 'scene_change' },
  ];

  it('maps frame results to chart points', () => {
    expect(buildTimelineData(frames)).toEqual([
      { t: 0.4, p: 0.3, frame: 4, flagged: false, selection: 'uniform' },
      { t: 4, p: 0.7, frame: 40, flagged: true, selection: 'scene_change' },
    ]);
  });

  it('never returns a zero-width axis and tolerates short duration metadata', () => {
    expect(timelineMaxTime([], undefined)).toBe(1);
    const points = buildTimelineData(frames);
    expect(timelineMaxTime(points, 3)).toBe(4);
    expect(timelineMaxTime(points, 10)).toBe(10);
  });
});

describe('signals', () => {
  const explanation = [
    { key: 'a', scored: true, value: -2.31, unit: 'log10 ratio' },
    { key: 'b', scored: false, value: null, unit: null },
    { key: 'c', scored: true, value: 3, unit: null },
  ];

  it('separates scored items from context-only items', () => {
    const { scored, context } = splitSignals(explanation);
    expect(scored.map((s) => s.key)).toEqual(['a', 'c']);
    expect(context.map((s) => s.key)).toEqual(['b']);
    expect(splitSignals(undefined)).toEqual({ scored: [], context: [] });
  });

  it('formats values with optional units', () => {
    expect(formatSignalValue(explanation[0])).toBe('-2.31 log10 ratio');
    expect(formatSignalValue(explanation[2])).toBe('3');
    expect(formatSignalValue(explanation[1])).toBeNull();
  });
});

describe('mediaStore', () => {
  beforeEach(forgetMedia);

  it('recalls only the media of the matching result', () => {
    const file = { name: 'a.mp4' };
    rememberMedia('r1', file, 'video');
    expect(recallMedia('r1')).toEqual({ resultId: 'r1', file, kind: 'video' });
    expect(recallMedia('r2')).toBeNull();
  });

  it('keeps only the most recent file and can be cleared', () => {
    rememberMedia('r1', { name: 'a' }, 'image');
    rememberMedia('r2', { name: 'b' }, 'image');
    expect(recallMedia('r1')).toBeNull();
    expect(recallMedia('r2')).not.toBeNull();
    forgetMedia();
    expect(recallMedia('r2')).toBeNull();
  });
});