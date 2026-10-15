import { formatBytes, formatDateTime, formatDuration } from '../utils/formatting.js';
import { TIER_LABEL } from '../utils/resultPresentation.js';

function mediaRows(result) {
  const m = result.metadata;
  if (result.media_type === 'video') {
    return [
      ['Resolution', `${m.width} × ${m.height}`],
      ['Megapixels', `${m.megapixels} MP`],
      ['Frame rate', `${m.fps} fps`],
      ['Duration', `${formatDuration(m.duration_sec)} (${m.duration_sec} s)`],
      ['Codec', m.codec ?? 'Unknown'],
      ['Frames reported by the container', m.frame_count_reported],
      ['Frames actually decoded', m.frames_decoded],
      ['File size', formatBytes(m.size_bytes)],
    ];
  }
  return [
    ['Format', m.format.toUpperCase()],
    ['Dimensions', `${m.width} × ${m.height}`],
    ['Megapixels', `${m.megapixels} MP`],
    ['Colour mode', m.color_mode],
    ['Transparency', m.has_alpha ? 'Yes (flattened onto white for analysis)' : 'No'],
    ['EXIF metadata', m.has_exif ? 'Present (contents are not read out)' : 'None found'],
    ['File size', formatBytes(m.size_bytes)],
  ];
}

function analysisRows(result) {
  return [
    ['Result ID', result.id],
    ['Analyzed at', formatDateTime(result.created_at)],
    ['Processing time', `${result.processing_time_ms} ms`],
    ['Model', `${result.model} v${result.detector_version}`],
    ['Detector tier', TIER_LABEL[result.detector_tier] ?? result.detector_tier],
    ['Score is calibrated', result.score_is_calibrated ? 'Yes' : 'No'],
  ];
}

function List({ rows }) {
  return (
    <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
      {rows.map(([label, value]) => (
        <div key={label} className="min-w-0">
          <dt className="text-muted">{label}</dt>
          <dd className="break-words font-medium">{String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function MetadataPanel({ result }) {
  return (
    <section aria-labelledby="metadata-title" className="rounded-2xl border bg-surface p-5">
      <h2 id="metadata-title" className="text-lg font-bold">
        Technical details
      </h2>

      <h3 className="mt-3 mb-2 text-sm font-semibold">
        {result.media_type === 'video' ? 'Video' : 'Image'}
      </h3>
      <List rows={mediaRows(result)} />

      <h3 className="mt-5 mb-2 text-sm font-semibold">Analysis</h3>
      <List rows={analysisRows(result)} />

      <p className="mt-4 text-xs text-muted">
        Metadata is shown for context only. It is not evidence of authenticity or of AI generation:
        platforms strip it from real files and it can be added to synthetic ones.
      </p>
    </section>
  );
}