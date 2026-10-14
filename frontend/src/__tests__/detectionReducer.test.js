import { describe, expect, it } from 'vitest';

import { detectionReducer, initialDetectionState } from '../hooks/useDetection.js';

const run = (...actions) => actions.reduce(detectionReducer, initialDetectionState);

describe('detectionReducer', () => {
  it('starts idle', () => {
    expect(initialDetectionState.status).toBe('idle');
  });

  it('start -> uploading with zero progress', () => {
    expect(run({ type: 'start' })).toMatchObject({ status: 'uploading', progress: 0 });
  });

  it('tracks upload progress, then moves to analyzing', () => {
    const uploading = run({ type: 'start' }, { type: 'progress', progress: 0.4 });
    expect(uploading.progress).toBe(0.4);

    const analyzing = detectionReducer(uploading, { type: 'uploaded' });
    expect(analyzing).toMatchObject({ status: 'analyzing', progress: null });
  });

  it('ignores progress events that arrive after the upload phase', () => {
    const state = run({ type: 'start' }, { type: 'uploaded' }, { type: 'progress', progress: 0.9 });
    expect(state).toMatchObject({ status: 'analyzing', progress: null });
  });

  it('ignores progress and uploaded events when idle', () => {
    expect(run({ type: 'progress', progress: 0.5 })).toBe(initialDetectionState);
    expect(run({ type: 'uploaded' })).toBe(initialDetectionState);
  });

  it('stores the result on success', () => {
    const result = { id: 'x'.repeat(32) };
    expect(run({ type: 'start' }, { type: 'success', result })).toMatchObject({
      status: 'success',
      result,
      error: null,
    });
  });

  it('stores the error on failure and clears it on the next start', () => {
    const error = new Error('boom');
    const failed = run({ type: 'start' }, { type: 'failure', error });
    expect(failed).toMatchObject({ status: 'error', error, result: null });
    expect(detectionReducer(failed, { type: 'start' }).error).toBeNull();
  });

  it('reset returns to idle', () => {
    expect(run({ type: 'start' }, { type: 'reset' })).toEqual(initialDetectionState);
  });
});