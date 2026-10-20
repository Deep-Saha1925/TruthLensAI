import { TriangleAlert } from 'lucide-react';

import { evaluationApplies, parseEvaluation } from '../utils/modelEvaluation.js';
import { TONE_CLASSES } from '../utils/resultPresentation.js';

/**
 * Measured performance of the model that produced this result, from /api/models.
 * Hidden when the server runs a different model than the one behind the result.
 */
export default function ModelPerformancePanel({ detector, result }) {
  if (!evaluationApplies(detector, result)) return null;
  const parsed = parseEvaluation(detector.evaluation);
  if (!parsed) return null;

  const measuredHere = detector.tier === 'TRAINED';

  return (
    <section aria-labelledby="performance-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="performance-title" className="text-lg font-bold">
        Measured performance of this model
      </h2>
      <p className="mt-1 text-sm text-muted">
        {measuredHere
          ? "Measured by this project's pipeline on a held-out test split of the training data. Performance on other generators, cameras, platforms and recompressed files is usually lower."
          : "Recorded in the model's metadata. This project did not measure it, and it may not apply to your images."}
      </p>

      {parsed.warning && (
        <p
          role="note"
          className={`mt-3 flex items-start gap-2 rounded-xl border p-3 text-sm ${TONE_CLASSES.warn}`}
        >
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <span>{parsed.warning}</span>
        </p>
      )}

      {parsed.headline.length > 0 && (
        <dl className="mt-4 grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
          {parsed.headline.map((metric) => (
            <div key={metric.key} className="flex items-baseline justify-between gap-3 border-b pb-2">
              <dt className="text-muted">{metric.label}</dt>
              <dd className="font-semibold tabular-nums">{metric.text}</dd>
            </div>
          ))}
        </dl>
      )}

      {parsed.robustness.length > 0 && (
        <div className="mt-5">
          <h3 className="text-sm font-semibold">How it holds up on modified test images</h3>
          <div className="mt-2 overflow-x-auto rounded-lg border">
            <table className="w-full text-left text-sm">
              <caption className="sr-only">Performance on degraded test images</caption>
              <thead className="bg-surface-2">
                <tr>
                  <th scope="col" className="px-3 py-2">Condition</th>
                  <th scope="col" className="px-3 py-2">ROC-AUC</th>
                  <th scope="col" className="px-3 py-2">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t">
                  <th scope="row" className="px-3 py-2 font-medium">Unmodified test images</th>
                  <td className="px-3 py-2 tabular-nums">{parsed.baseline.auc}</td>
                  <td className="px-3 py-2 tabular-nums">{parsed.baseline.accuracy}</td>
                </tr>
                {parsed.robustness.map((row) => (
                  <tr key={row.key} className="border-t">
                    <th scope="row" className="px-3 py-2 font-medium">{row.label}</th>
                    <td className="px-3 py-2 tabular-nums">{row.auc}</td>
                    <td className="px-3 py-2 tabular-nums">{row.accuracy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">
            A large drop from the first row means the model depends on fine detail that
            compression, resizing or blur destroys, which is common for real-world sharing.
          </p>
        </div>
      )}

      {(parsed.notes.length > 0 || parsed.other.length > 0) && (
        <dl className="mt-5 space-y-2 text-sm">
          {[...parsed.notes, ...parsed.other].map((note) => (
            <div key={note.key}>
              <dt className="text-muted">{note.label}</dt>
              <dd className="break-words font-medium">{note.text}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}