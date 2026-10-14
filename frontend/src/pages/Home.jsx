import { Activity, Film, Image as ImageIcon, Layers, UploadCloud } from 'lucide-react';
import { useCallback, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

import DisclaimerNotice from '../components/DisclaimerNotice.jsx';
import ErrorMessage from '../components/ErrorMessage.jsx';
import LoadingState from '../components/LoadingState.jsx';
import MediaPreview from '../components/MediaPreview.jsx';
import UploadZone from '../components/UploadZone.jsx';
import { useDetection } from '../hooks/useDetection.js';
import { isRetryable } from '../services/api.js';
import { formatBytes, formatDuration } from '../utils/formatting.js';
import { detectMediaKind, validateFile, validateVideoDuration } from '../utils/validation.js';

const STEPS = [
  {
    Icon: UploadCloud,
    title: '1. Upload',
    text: 'Your file is checked for type, size and integrity, then decoded on the server. It is deleted right after analysis.',
  },
  {
    Icon: Layers,
    title: '2. Analyze',
    text: 'Images are scored by the active detector. Videos are sampled at evenly spaced frames plus scene changes, and each sampled frame is scored.',
  },
  {
    Icon: Activity,
    title: '3. Interpret',
    text: 'You get probabilities, an "uncertain" band for ambiguous cases, and a label saying how trustworthy the detector itself is.',
  },
];

export default function Home({ serverInfo }) {
  const { limits, limitsFromServer } = serverInfo;
  const navigate = useNavigate();
  const { status, progress, error: analysisError, analyze, cancel, reset: resetDetection } =
    useDetection();

  const [file, setFile] = useState(null);
  const [mediaInfo, setMediaInfo] = useState(null);
  const [error, setError] = useState(null);

  // 'success' is included so the file card does not flash back before navigation.
  const showLoading = status === 'uploading' || status === 'analyzing' || status === 'success';
  const kind = file ? detectMediaKind(file.name, limits) : null;
  const durationError = kind === 'video' ? validateVideoDuration(mediaInfo?.duration, limits) : null;

  const handleFilesSelected = useCallback(
    (files) => {
      resetDetection();
      setMediaInfo(null);
      if (files.length > 1) {
        setFile(null);
        setError('Please choose one file at a time.');
        return;
      }
      const result = validateFile(files[0], limits);
      if (!result.ok) {
        setFile(null);
        setError(result.message);
        return;
      }
      setError(null);
      setFile(files[0]);
    },
    [limits, resetDetection],
  );

  const reset = () => {
    resetDetection();
    setFile(null);
    setMediaInfo(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file || !kind) return;
    const result = await analyze(file, kind); // null when failed, cancelled or superseded
    if (result) navigate(`/results/${result.id}`, { state: { result } });
  };

  return (
    <div className="mx-auto max-w-5xl px-4 pb-16">
      <section className="pt-10 sm:pt-16" aria-labelledby="hero-title">
        <p className="text-sm font-medium text-accent">Probabilistic Media Authenticity Analysis</p>
        <h1 id="hero-title" className="mt-2 text-3xl font-bold tracking-tight sm:text-5xl">
          TruthLens AI
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-muted">
          AI-generated media is imagery created or altered by generative models, from fully
          synthetic images to face-swapped video. It is getting harder to recognise by eye. Upload
          an image or video to get an <em>estimate</em> of how likely it is to be synthetic or
          manipulated, together with an honest account of how much that estimate can be trusted.
        </p>
      </section>

      <section className="mt-8 space-y-4" aria-labelledby="upload-title">
        <h2 id="upload-title" className="sr-only">
          Analyze a file
        </h2>

        {error && (
          <ErrorMessage
            title="Cannot use this file"
            message={error}
            onDismiss={() => setError(null)}
          />
        )}

        {status === 'error' && analysisError && (
          <ErrorMessage
            title="Analysis failed"
            message={analysisError.message}
            requestId={analysisError.requestId}
            onDismiss={resetDetection}
            onRetry={isRetryable(analysisError) ? handleAnalyze : undefined}
          />
        )}

        {showLoading ? (
          <LoadingState
            phase={status === 'uploading' ? 'uploading' : 'analyzing'}
            progress={progress}
            kind={kind}
            fileName={file?.name}
            onCancel={cancel}
          />
        ) : !file ? (
          <UploadZone limits={limits} onFilesSelected={handleFilesSelected} />
        ) : (
          <div className="rounded-2xl border bg-surface p-4 sm:p-6">
            <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_16rem]">
              <MediaPreview file={file} kind={kind} onInfo={setMediaInfo} />

              <div className="flex flex-col gap-4">
                <div>
                  <h3 className="flex items-center gap-2 font-semibold">
                    {kind === 'image' ? (
                      <ImageIcon className="h-4 w-4" aria-hidden="true" />
                    ) : (
                      <Film className="h-4 w-4" aria-hidden="true" />
                    )}
                    Selected {kind}
                  </h3>
                  <dl className="mt-3 space-y-2 text-sm">
                    <div>
                      <dt className="text-muted">Name</dt>
                      <dd className="break-all font-medium">{file.name}</dd>
                    </div>
                    <div>
                      <dt className="text-muted">Size</dt>
                      <dd className="font-medium">{formatBytes(file.size)}</dd>
                    </div>
                    {mediaInfo?.width > 0 && (
                      <div>
                        <dt className="text-muted">Dimensions</dt>
                        <dd className="font-medium">
                          {mediaInfo.width} × {mediaInfo.height}
                        </dd>
                      </div>
                    )}
                    {Number.isFinite(mediaInfo?.duration) && (
                      <div>
                        <dt className="text-muted">Duration</dt>
                        <dd className="font-medium">{formatDuration(mediaInfo.duration)}</dd>
                      </div>
                    )}
                  </dl>
                </div>

                {durationError && <ErrorMessage title="Video too long" message={durationError} />}

                <div className="mt-auto flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={Boolean(durationError)}
                    className="rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Analyze Media
                  </button>
                  <button
                    type="button"
                    onClick={reset}
                    className="rounded-lg border px-5 py-2.5 font-medium transition-colors hover:bg-surface-2"
                  >
                    Choose a different file
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        <DisclaimerNotice />
      </section>

      <section className="mt-14" aria-labelledby="how-title">
        <h2 id="how-title" className="text-2xl font-bold">
          How detection works
        </h2>
        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          {STEPS.map(({ Icon, title, text }) => (
            <div key={title} className="rounded-2xl border bg-surface p-5">
              <Icon className="h-6 w-6 text-accent" aria-hidden="true" />
              <h3 className="mt-3 font-semibold">{title}</h3>
              <p className="mt-2 text-sm text-muted">{text}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 text-sm text-muted">
          No detector can reliably identify every AI-generated image or video. Compression,
          resizing and new generators all reduce accuracy.{' '}
          <Link to="/about" className="font-medium text-accent underline underline-offset-2">
            Read the limitations
          </Link>
          .
        </p>
      </section>

      <section className="mt-14" aria-labelledby="limits-title">
        <h2 id="limits-title" className="text-2xl font-bold">
          Supported formats and limits
        </h2>
        {!limitsFromServer && (
          <p className="mt-2 text-sm text-muted">
            Showing default limits; the server's actual limits could not be loaded.
          </p>
        )}
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <div className="rounded-2xl border bg-surface p-5">
            <h3 className="flex items-center gap-2 font-semibold">
              <ImageIcon className="h-4 w-4" aria-hidden="true" /> Images
            </h3>
            <p className="mt-2 text-sm text-muted">
              {limits.image_extensions.join(', ')}. Up to {limits.max_image_size_mb} MB.
            </p>
          </div>
          <div className="rounded-2xl border bg-surface p-5">
            <h3 className="flex items-center gap-2 font-semibold">
              <Film className="h-4 w-4" aria-hidden="true" /> Videos
            </h3>
            <p className="mt-2 text-sm text-muted">
              {limits.video_extensions.join(', ')}. Up to {limits.max_video_size_mb} MB and{' '}
              {limits.max_video_duration_sec} seconds; up to {limits.max_video_frames_analyzed}{' '}
              frames are analyzed.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}