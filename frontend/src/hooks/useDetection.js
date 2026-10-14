import { useCallback, useEffect, useReducer, useRef } from 'react';

import { detectImage, detectVideo, isCancelled } from '../services/api.js';

export const initialDetectionState = {
  status: 'idle', // idle | uploading | analyzing | success | error
  progress: null, // 0..1 while uploading, null otherwise
  result: null,
  error: null,
};

export function detectionReducer(state, action) {
  switch (action.type) {
    case 'start':
      return { status: 'uploading', progress: 0, result: null, error: null };
    case 'progress':
      // Progress events can arrive after the upload phase ended: ignore them.
      return state.status === 'uploading' ? { ...state, progress: action.progress } : state;
    case 'uploaded':
      return state.status === 'uploading' ? { ...state, status: 'analyzing', progress: null } : state;
    case 'success':
      return { status: 'success', progress: null, result: action.result, error: null };
    case 'failure':
      return { status: 'error', progress: null, result: null, error: action.error };
    case 'reset':
      return initialDetectionState;
    default:
      return state;
  }
}

/**
 * Drives one upload + analysis at a time.
 * analyze() resolves with the result, or null if it failed, was cancelled or was superseded.
 */
export function useDetection() {
  const [state, dispatch] = useReducer(detectionReducer, initialDetectionState);
  const requestId = useRef(0);
  const controller = useRef(null);

  const invalidate = useCallback(() => {
    requestId.current += 1; // any response still in flight is now stale
    controller.current?.abort();
    controller.current = null;
  }, []);

  const analyze = useCallback(
    async (file, kind) => {
      invalidate();
      const id = requestId.current;
      const abort = new AbortController();
      controller.current = abort;
      dispatch({ type: 'start' });

      const send = kind === 'video' ? detectVideo : detectImage;
      try {
        const result = await send(file, {
          signal: abort.signal,
          onUploadProgress: (event) => {
            if (id !== requestId.current || event.progress == null) return;
            dispatch({ type: 'progress', progress: event.progress });
            // All bytes sent: from here on the wait is the server analyzing.
            if (event.progress >= 1) dispatch({ type: 'uploaded' });
          },
        });
        if (id !== requestId.current) return null;
        dispatch({ type: 'success', result });
        return result;
      } catch (error) {
        if (id !== requestId.current || isCancelled(error)) return null;
        dispatch({ type: 'failure', error });
        return null;
      }
    },
    [invalidate],
  );

  const reset = useCallback(() => {
    invalidate();
    dispatch({ type: 'reset' });
  }, [invalidate]);

  // Leaving the page abandons any request in flight.
  useEffect(() => invalidate, [invalidate]);

  return { ...state, analyze, cancel: reset, reset };
}