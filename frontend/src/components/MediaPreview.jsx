import { FileQuestion } from 'lucide-react';
import { useEffect, useState } from 'react';

/**
 * Local preview of the selected file via an object URL (nothing is uploaded).
 * Reports intrinsic size/duration through onInfo once the browser has decoded the header.
 * `videoRef` lets a parent seek the player (used by the results dashboard).
 */
export default function MediaPreview({ file, kind, onInfo, videoRef }) {
  const [url, setUrl] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    setFailed(false);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  if (!url) return <div className="aspect-video animate-pulse rounded-xl bg-surface-2" aria-hidden="true" />;

  if (failed) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-2 rounded-xl bg-surface-2 p-4 text-center text-sm text-muted">
        <FileQuestion className="h-8 w-8" aria-hidden="true" />
        <p>
          Your browser cannot preview this {kind}. You can still analyze it; the server decodes the
          file itself.
        </p>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center overflow-hidden rounded-xl bg-surface-2">
      {kind === 'image' ? (
        <img
          src={url}
          alt={`Preview of the selected image ${file.name}`}
          className="max-h-[420px] w-full object-contain"
          onLoad={(event) =>
            onInfo?.({ width: event.target.naturalWidth, height: event.target.naturalHeight })
          }
          onError={() => setFailed(true)}
        />
      ) : (
        <video
          ref={videoRef}
          src={url}
          controls
          playsInline
          preload="metadata"
          aria-label={`Preview of the selected video ${file.name}`}
          className="max-h-[420px] w-full"
          onLoadedMetadata={(event) =>
            onInfo?.({
              width: event.target.videoWidth,
              height: event.target.videoHeight,
              duration: event.target.duration,
            })
          }
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}