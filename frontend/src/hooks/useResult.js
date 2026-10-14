import { useCallback, useEffect, useState } from 'react';

import { getResult, isCancelled } from '../services/api.js';

/**
 * Loads one result. Uses `initial` (passed through router state right after an analysis)
 * when it matches the id; otherwise fetches GET /api/results/{id}, which is what happens
 * after a page refresh or when someone opens a link.
 */
export function useResult(resultId, initial = null) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState(() =>
    initial?.id === resultId
      ? { status: 'ready', result: initial, error: null }
      : { status: 'loading', result: null, error: null },
  );

  useEffect(() => {
    if (initial?.id === resultId) {
      setState({ status: 'ready', result: initial, error: null });
      return undefined;
    }

    const controller = new AbortController();
    setState({ status: 'loading', result: null, error: null });
    getResult(resultId, { signal: controller.signal })
      .then((result) => setState({ status: 'ready', result, error: null }))
      .catch((error) => {
        if (isCancelled(error)) return;
        setState({ status: 'error', result: null, error });
      });
    return () => controller.abort();
  }, [resultId, initial, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return { ...state, retry };
}