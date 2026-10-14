/**
 * Remembers the file behind the most recent result, in browser memory only.
 *
 * The server deletes uploads right after analysis, so this is the only way the dashboard
 * can show the original. It is lost on refresh or in another tab, which the UI explains.
 */
let current = null; // { resultId, file, kind }

export function rememberMedia(resultId, file, kind) {
  current = { resultId, file, kind };
}

export function recallMedia(resultId) {
  return current && current.resultId === resultId ? current : null;
}

export function forgetMedia() {
  current = null;
}