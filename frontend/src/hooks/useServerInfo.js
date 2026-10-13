import { useEffect, useState } from 'react';

import { getModels, isCancelled } from '../services/api.js';
import { DEFAULT_LIMITS } from '../utils/validation.js';

/**
 * Loads limits and detector info from GET /api/models once.
 * status: 'loading' | 'ready' | 'unreachable'
 * `limitsFromServer` is false while the built-in defaults are in use.
 */
export function useServerInfo() {
  const [state, setState] = useState({
    status: 'loading',
    limits: DEFAULT_LIMITS,
    limitsFromServer: false,
    detector: null,
    uncertainBand: null,
    error: null,
  });

  useEffect(() => {
    const controller = new AbortController();

    getModels({ signal: controller.signal })
      .then((data) =>
        setState({
          status: 'ready',
          limits: data.limits,
          limitsFromServer: true,
          detector: data.active_detector,
          uncertainBand: data.uncertain_band,
          error: null,
        }),
      )
      .catch((error) => {
        if (isCancelled(error)) return;
        setState((previous) => ({ ...previous, status: 'unreachable', error: error.message }));
      });

    return () => controller.abort();
  }, []);

  return state;
}