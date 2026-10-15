import { formatPercent } from '../utils/formatting.js';

const ROWS = [
  ['trimmed_mean', 'Trimmed mean', 'Typical level after dropping the highest and lowest 10% of frames', 'percent'],
  ['top_quarter_mean', 'Top-quarter mean', 'Average of the highest-scoring 25% of frames', 'percent'],
  ['mean', 'Plain mean', 'Simple average of all frame scores', 'percent'],
  ['minimum', 'Lowest frame score', '', 'percent'],
  ['maximum', 'Highest frame score', '', 'percent'],
  ['std_dev', 'Spread (std. deviation)', 'Large values mean the frames disagree', 'number'],
  ['flagged_fraction', 'Share of frames flagged', 'At or above the synthetic threshold', 'percent'],
];

export default function AggregationPanel({ sampling, aggregation }) {
  return (
    <section aria-labelledby="aggregation-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="aggregation-title" className="text-lg font-bold">
        How the video score was built
      </h2>

      <h3 className="mt-3 text-sm font-semibold">Sampling</h3>
      <p className="mt-1 text-sm text-muted">
        {sampling.frames_analyzed} of {sampling.frames_decoded} decoded frames were analyzed (
        {formatPercent(sampling.coverage_fraction, 1)} coverage): {sampling.uniform_frames} evenly
        spaced and {sampling.scene_change_frames} at detected scene changes. {sampling.description}
      </p>

      <h3 className="mt-5 text-sm font-semibold">Aggregation</h3>
      <p className="mt-1 text-sm text-muted">
        Video score = {aggregation.method}. {aggregation.rule_note}
      </p>

      <div className="mt-3 overflow-x-auto rounded-lg border">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Frame score statistics</caption>
          <thead className="bg-surface-2">
            <tr>
              <th scope="col" className="px-3 py-2">Statistic</th>
              <th scope="col" className="px-3 py-2">Value</th>
              <th scope="col" className="hidden px-3 py-2 sm:table-cell">Meaning</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([key, label, meaning, kind]) => (
              <tr key={key} className="border-t">
                <th scope="row" className="px-3 py-2 font-medium">{label}</th>
                <td className="px-3 py-2 tabular-nums">
                  {kind === 'percent' ? formatPercent(aggregation[key], 1) : aggregation[key].toFixed(3)}
                </td>
                <td className="hidden px-3 py-2 text-muted sm:table-cell">{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}