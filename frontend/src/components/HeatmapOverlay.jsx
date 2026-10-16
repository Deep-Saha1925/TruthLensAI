import { useEffect, useId, useState } from 'react';

import {
  DEFAULT_HEATMAP_OPACITY_PERCENT,
  HEATMAP_LEGEND_GRADIENT,
  percentToOpacity,
} from '../utils/heatmap.js';

/**
 * Shows the server's heatmap PNG over the original image (kept in browser memory).
 * Without the original (after a refresh) the heatmap is shown on its own.
 */
export default function HeatmapOverlay({ heatmap, file }) {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [visible, setVisible] = useState(true);
  const [percent, setPercent] = useState(DEFAULT_HEATMAP_OPACITY_PERCENT);
  const sliderId = useId();

  useEffect(() => {
    if (!file) {
      setOriginalUrl(null);
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setOriginalUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const opacity = visible ? percentToOpacity(percent) : 0;
  const hasOriginal = Boolean(file && originalUrl);

  return (
    <div>
      <div className="flex justify-center overflow-hidden rounded-xl bg-surface-2">
        {hasOriginal ? (
          <div className="relative inline-block max-w-full">
            <img
              src={originalUrl}
              alt="The analyzed image"
              className="block max-h-[420px] max-w-full"
            />
            <img
              src={heatmap.data_url}
              alt=""
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full"
              style={{ opacity }}
            />
          </div>
        ) : (
          <img
            src={heatmap.data_url}
            alt="Heatmap of the regions that most influenced the model's score"
            className="block max-h-[420px] max-w-full"
            style={{ opacity }}
          />
        )}
      </div>

      {!hasOriginal && (
        <p className="mt-2 text-xs text-muted">
          The original image is not available in this browser session, so the heatmap is shown
          without it.
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={visible}
            onChange={(event) => setVisible(event.target.checked)}
          />
          Show heatmap
        </label>
        <label htmlFor={sliderId} className="flex items-center gap-2">
          Opacity
          <input
            id={sliderId}
            type="range"
            min="0"
            max="100"
            step="5"
            value={percent}
            disabled={!visible}
            aria-valuetext={`${percent}%`}
            onChange={(event) => setPercent(Number(event.target.value))}
          />
          <span className="w-10 tabular-nums">{percent}%</span>
        </label>
      </div>

      <div className="mt-3">
        <div
          className="h-3 rounded-full"
          style={{ backgroundImage: HEATMAP_LEGEND_GRADIENT }}
          aria-hidden="true"
        />
        <div className="mt-1 flex justify-between text-xs text-muted">
          <span>Lower influence</span>
          <span>Higher influence</span>
        </div>
      </div>
      <p className="mt-2 text-xs text-muted">
        Method: Grad-CAM on the {heatmap.target}. Warm colours mark regions that pushed the score
        toward "synthetic". The map is coarse and approximate.
      </p>
    </div>
  );
}