import { FileQuestion, FlaskConical, ShieldCheck, TriangleAlert } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import AggregationPanel from './AggregationPanel.jsx';
import ClassificationBadge from './ClassificationBadge.jsx';
import ConfidenceMeter from './ConfidenceMeter.jsx';
import DisclaimerNotice from './DisclaimerNotice.jsx';
import ExplainabilityPanel from './ExplainabilityPanel.jsx';
import MediaPreview from './MediaPreview.jsx';
import MetadataPanel from './MetadataPanel.jsx';
import ProbabilityChart from './ProbabilityChart.jsx';
import RiskBadge from './RiskBadge.jsx';
import SuspiciousFrames from './SuspiciousFrames.jsx';
import VideoTimeline from './VideoTimeline.jsx';
import { formatPercent } from '../utils/formatting.js';
import { TIER_LABEL, TIER_TONE, TONE_CLASSES } from '../utils/resultPresentation.js';

function TierNotice({ tier, text }) {
  const Icon = tier === 'DEMO_HEURISTIC' ? FlaskConical : ShieldCheck;
  return (
    <div
      role="note"
      className={`flex items-start gap-3 rounded-xl border p-4 text-sm ${TONE_CLASSES[TIER_TONE[tier] ?? 'warn']}`}
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <p>
        <strong>{TIER_LABEL[tier] ?? 'Detector'}:</strong> {text}
      </p>
    </div>
  );
}

function Stat({ label, children }) {
  return (
    <div>
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="mt-0.5 font-medium">{children}</dd>
    </div>
  );
}

/**
 * The results dashboard for one image or video result.
 * `media` is { file, kind } when the original is still in browser memory, otherwise null.
 */
export default function DetectionResult({ result, media }) {
  const isVideo = result.media_type === 'video';
  const videoRef = useRef(null);
  const headingRef = useRef(null);
  const [playerReady, setPlayerReady] = useState(false);

  // Move focus to the page heading after navigating here.
  useEffect(() => {
    headingRef.current?.focus();
  }, [result.id]);

  const canSeek = isVideo && Boolean(media) && playerReady;

  const seekTo = useCallback((seconds) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, seconds);
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    video.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
  }, []);

  const scorePercent = formatPercent(result.ai_probability);

  return (
    <div className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <header>
        <p className="text-sm font-medium text-accent">{isVideo ? 'Video' : 'Image'} analysis</p>
        <h1
          ref={headingRef}
          tabIndex={-1}
          className="mt-1 text-3xl font-bold tracking-tight outline-none"
        >
          Analysis result
        </h1>
      </header>

      <TierNotice tier={result.detector_tier} text={result.tier_notice} />

      <div className="grid gap-6 md:grid-cols-2">
        <section aria-labelledby="media-title" className="rounded-2xl border bg-surface p-4">
          <h2 id="media-title" className="mb-3 font-semibold">
            Analyzed {isVideo ? 'video' : 'image'}
          </h2>
          {media ? (
            <MediaPreview
              file={media.file}
              kind={media.kind}
              videoRef={videoRef}
              onInfo={() => setPlayerReady(true)}
            />
          ) : (
            <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-xl bg-surface-2 p-4 text-center text-sm text-muted">
              <FileQuestion className="h-8 w-8" aria-hidden="true" />
              <p>
                The original is not shown. Uploaded files are deleted from the server after
                analysis, and this page was opened without the file in this browser session.
              </p>
            </div>
          )}
        </section>

        <section aria-labelledby="summary-title" className="space-y-5 rounded-2xl border bg-surface p-5">
          <h2 id="summary-title" className="sr-only">
            Summary
          </h2>
          <ClassificationBadge
            classification={result.classification}
            label={result.classification_label}
            large
          />
          <p className="text-muted">
            Estimated score for AI generation or manipulation:{' '}
            <strong className="text-fg">{scorePercent}</strong>
            {!result.score_is_calibrated && ' (raw score, not a calibrated probability)'}.
          </p>

          <ConfidenceMeter
            confidence={result.confidence}
            aiProbability={result.ai_probability}
            calibrated={result.score_is_calibrated}
          />

          <dl className="grid grid-cols-2 gap-4 border-t pt-4">
            <Stat label="Risk level">
              <RiskBadge level={result.risk_level} />
            </Stat>
            <Stat label="Processing time">{result.processing_time_ms} ms</Stat>
            <Stat label="Model">
              {result.model} v{result.detector_version}
            </Stat>
            {isVideo && (
              <Stat label="Frames analyzed">
                {result.frames_analyzed}
                <span className="text-sm text-muted">
                  {' '}
                  of {result.sampling.frames_decoded}
                </span>
              </Stat>
            )}
          </dl>
        </section>
      </div>

      <ProbabilityChart
        aiProbability={result.ai_probability}
        uncertainBand={result.uncertain_band}
        calibrated={result.score_is_calibrated}
      />

      {isVideo && (
        <>
          {result.warnings?.length > 0 && (
            <section
              aria-labelledby="warnings-title"
              className={`rounded-xl border p-4 ${TONE_CLASSES.warn}`}
            >
              <h2 id="warnings-title" className="flex items-center gap-2 font-semibold">
                <TriangleAlert className="h-5 w-5" aria-hidden="true" />
                Read before relying on this video result
              </h2>
              <ul className="mt-2 list-disc space-y-1 pl-6 text-sm">
                {result.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </section>
          )}

          <VideoTimeline
            frames={result.frame_results}
            uncertainBand={result.uncertain_band}
            durationSec={result.metadata.duration_sec}
            canSeek={canSeek}
            onSeek={seekTo}
          />
          <SuspiciousFrames
            frames={result.highest_scoring_frames}
            threshold={result.aggregation.flag_threshold}
            canSeek={canSeek}
            onSeek={seekTo}
          />
          <AggregationPanel sampling={result.sampling} aggregation={result.aggregation} />
        </>
      )}

      <ExplainabilityPanel
        explanation={result.explanation}
        notice={result.explanation_notice}
        tier={result.detector_tier}
        mediaType={result.media_type}
        heatmap={result.heatmap ?? null}
        heatmapNotice={result.heatmap_notice}
        originalFile={media && media.kind === 'image' ? media.file : null}
      />

      <MetadataPanel result={result} />

      <details className="rounded-2xl border bg-surface p-4">
        <summary className="cursor-pointer font-medium">Raw response (JSON)</summary>
        <pre className="mt-3 max-h-96 overflow-auto rounded-lg bg-surface-2 p-3 text-xs">
          {JSON.stringify(
            result.heatmap
              ? { ...result, heatmap: { ...result.heatmap, data_url: '(PNG data omitted here)' } }
              : result,
            null,
            2,
          )}
        </pre>
      </details>

      <DisclaimerNotice />

      <Link
        to="/"
        className="inline-block rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity hover:opacity-90"
      >
        Analyze Another File
      </Link>
    </div>
  );
}