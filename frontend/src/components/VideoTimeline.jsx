import { useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { formatDuration, formatPercent, formatTimestamp } from '../utils/formatting.js';
import { buildTimelineData, timelineMaxTime } from '../utils/resultPresentation.js';

const AXIS_TICK = { fill: 'var(--muted)', fontSize: 12 };

/** Shape and size carry the meaning: square = scene-change frame, larger = flagged. */
function FrameDot({ cx, cy, payload }) {
  if (cx == null || cy == null || !payload) return null;
  const fill = payload.flagged ? 'var(--flag)' : 'var(--accent)';
  const common = { fill, stroke: 'var(--surface)', strokeWidth: 1.5 };
  const key = `dot-${payload.frame}`;

  if (payload.selection === 'scene_change') {
    const half = payload.flagged ? 6 : 5;
    return <rect key={key} x={cx - half} y={cy - half} width={half * 2} height={half * 2} {...common} />;
  }
  return <circle key={key} cx={cx} cy={cy} r={payload.flagged ? 6 : 4} {...common} />;
}

function FrameTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded-lg border bg-surface p-3 text-xs shadow-lg">
      <p className="font-semibold">
        Frame {point.frame} · {formatTimestamp(point.t)}
      </p>
      <p>Score: {formatPercent(point.p)}</p>
      <p className="text-muted">
        {point.selection === 'scene_change' ? 'Chosen at a scene change' : 'Evenly spaced sample'}
      </p>
      {point.flagged && <p className="font-medium">At or above the synthetic threshold</p>}
    </div>
  );
}

function LegendMark({ children }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" className="shrink-0">
      {children}
    </svg>
  );
}

export default function VideoTimeline({ frames, uncertainBand, durationSec, canSeek, onSeek }) {
  const [low, high] = uncertainBand;
  const data = useMemo(() => buildTimelineData(frames), [frames]);
  const maxTime = timelineMaxTime(data, durationSec);

  const handleChartClick = (state) => {
    const point = state?.activePayload?.[0]?.payload;
    if (point && canSeek) onSeek(point.t);
  };

  return (
    <section aria-labelledby="timeline-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="timeline-title" className="text-lg font-bold">
        Detection timeline
      </h2>
      <p className="mt-1 text-sm text-muted">
        Each marker is one analyzed frame, scored on its own. Frames between markers were not
        analyzed.{' '}
        {canSeek
          ? 'Click a marker to jump to that moment in the video above.'
          : 'Jumping to a moment needs the original video, which is only available in the browser session that uploaded it.'}
      </p>

      <div
        className="mt-4 h-72 w-full"
        role="img"
        aria-label={`Line chart of the AI score for ${data.length} analyzed frames over time. A table with the same values is available below.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 16, bottom: 20, left: 0 }}
            onClick={handleChartClick}
            style={{ cursor: canSeek ? 'pointer' : 'default' }}
          >
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, maxTime]}
              tickFormatter={formatDuration}
              tick={AXIS_TICK}
              stroke="var(--border)"
              label={{
                value: 'Time in video (m:ss)',
                position: 'insideBottom',
                offset: -12,
                fill: 'var(--muted)',
                fontSize: 12,
              }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(value) => `${Math.round(value * 100)}%`}
              tick={AXIS_TICK}
              stroke="var(--border)"
              width={44}
            />
            <ReferenceArea
              y1={low}
              y2={high}
              fill="var(--warn-border)"
              fillOpacity={0.18}
              label={{
                value: 'Uncertain band',
                position: 'insideBottomRight',
                fill: 'var(--muted)',
                fontSize: 11,
              }}
            />
            <ReferenceLine
              y={high}
              stroke="var(--flag)"
              strokeDasharray="5 4"
              label={{
                value: 'Synthetic threshold',
                position: 'insideTopRight',
                fill: 'var(--muted)',
                fontSize: 11,
              }}
            />
            <Tooltip content={<FrameTooltip />} />
            <Line
              type="linear"
              dataKey="p"
              stroke="var(--accent)"
              strokeWidth={2}
              dot={<FrameDot />}
              activeDot={{ r: 8 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted">
        <li className="flex items-center gap-1.5">
          <LegendMark>
            <circle cx="7" cy="7" r="4" fill="var(--accent)" />
          </LegendMark>
          Evenly spaced frame
        </li>
        <li className="flex items-center gap-1.5">
          <LegendMark>
            <rect x="2" y="2" width="10" height="10" fill="var(--accent)" />
          </LegendMark>
          Scene-change frame
        </li>
        <li className="flex items-center gap-1.5">
          <LegendMark>
            <circle cx="7" cy="7" r="6" fill="var(--flag)" />
          </LegendMark>
          Larger red marker: at or above the threshold
        </li>
      </ul>

      <details className="mt-4">
        <summary className="cursor-pointer text-sm font-medium">
          Show frame scores as a table
        </summary>
        <div className="mt-3 max-h-72 overflow-auto rounded-lg border">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">AI score for each analyzed frame</caption>
            <thead className="bg-surface-2">
              <tr>
                <th scope="col" className="px-3 py-2">Frame</th>
                <th scope="col" className="px-3 py-2">Time</th>
                <th scope="col" className="px-3 py-2">Score</th>
                <th scope="col" className="px-3 py-2">Sampled</th>
                <th scope="col" className="px-3 py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {frames.map((frame) => (
                <tr key={frame.frame_number} className="border-t">
                  <td className="px-3 py-2 tabular-nums">{frame.frame_number}</td>
                  <td className="px-3 py-2 tabular-nums">{formatTimestamp(frame.timestamp_sec)}</td>
                  <td className="px-3 py-2 tabular-nums">{formatPercent(frame.ai_probability)}</td>
                  <td className="px-3 py-2">
                    {frame.selection === 'scene_change' ? 'Scene change' : 'Evenly spaced'}
                  </td>
                  <td className="px-3 py-2">
                    {frame.flagged ? 'At or above threshold' : 'Below threshold'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}