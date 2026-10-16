import { describe, expect, it } from 'vitest';

import {
  DEFAULT_HEATMAP_OPACITY_PERCENT,
  HEATMAP_LEGEND_GRADIENT,
  heatmapAspectRatio,
  percentToOpacity,
} from '../utils/heatmap.js';

describe('percentToOpacity', () => {
  it('maps the slider range to CSS opacity', () => {
    expect(percentToOpacity(0)).toBe(0);
    expect(percentToOpacity(60)).toBeCloseTo(0.6);
    expect(percentToOpacity(100)).toBe(1);
  });

  it('clamps out-of-range values and falls back for invalid input', () => {
    expect(percentToOpacity(-20)).toBe(0);
    expect(percentToOpacity(250)).toBe(1);
    expect(percentToOpacity('abc')).toBeCloseTo(DEFAULT_HEATMAP_OPACITY_PERCENT / 100);
    expect(percentToOpacity(undefined)).toBeCloseTo(DEFAULT_HEATMAP_OPACITY_PERCENT / 100);
  });
});

describe('heatmap constants', () => {
  it('builds an aspect ratio string', () => {
    expect(heatmapAspectRatio({ width: 256, height: 192 })).toBe('256 / 192');
  });

  it('uses the five colour stops of the backend colour map', () => {
    expect(HEATMAP_LEGEND_GRADIENT.match(/#[0-9a-f]{6}/gi)).toEqual([
      '#313695',
      '#4575b4',
      '#fee090',
      '#f46d43',
      '#a50026',
    ]);
  });
});