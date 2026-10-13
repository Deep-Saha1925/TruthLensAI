import { formatBytes, formatDuration } from './formatting.js';

/** Used until /api/models answers (or if it never does). The server remains the authority. */
export const DEFAULT_LIMITS = {
  max_image_size_mb: 10,
  max_video_size_mb: 100,
  max_video_duration_sec: 60,
  max_video_frames_analyzed: 24,
  image_extensions: ['.jpeg', '.jpg', '.png', '.webp'],
  video_extensions: ['.avi', '.mkv', '.mov', '.mp4', '.webm'],
};

// Browsers and operating systems often send these for perfectly valid files.
const GENERIC_MIME_TYPES = new Set(['', 'application/octet-stream', 'binary/octet-stream']);
const BYTES_PER_MB = 1024 * 1024;

export function getExtension(filename = '') {
  const base = filename.split(/[\\/]/).pop() ?? '';
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(dot).toLowerCase() : '';
}

/** 'image' | 'video' | null, decided by extension only. */
export function detectMediaKind(filename, limits = DEFAULT_LIMITS) {
  const extension = getExtension(filename);
  if (limits.image_extensions.includes(extension)) return 'image';
  if (limits.video_extensions.includes(extension)) return 'video';
  return null;
}

const failure = (code, message) => ({ ok: false, code, message });

/**
 * Pre-flight check before uploading. A convenience only: the server repeats every
 * check, including real content sniffing and decoding.
 */
export function validateFile(file, limits = DEFAULT_LIMITS) {
  if (!file) return failure('NO_FILE', 'Please choose a file.');

  const kind = detectMediaKind(file.name, limits);
  if (!kind) {
    const supported = [...limits.image_extensions, ...limits.video_extensions].join(', ');
    return failure('UNSUPPORTED_EXTENSION', `Unsupported file type. Supported formats: ${supported}.`);
  }

  const mime = (file.type || '').toLowerCase();
  if (!GENERIC_MIME_TYPES.has(mime) && !mime.startsWith(`${kind}/`)) {
    return failure('MIME_MISMATCH', "The file's declared type does not match its extension.");
  }

  if (file.size === 0) return failure('EMPTY_FILE', 'The file is empty.');

  const maxMb = kind === 'image' ? limits.max_image_size_mb : limits.max_video_size_mb;
  if (file.size > maxMb * BYTES_PER_MB) {
    return failure(
      'FILE_TOO_LARGE',
      `This file is ${formatBytes(file.size)}. The limit for ${kind}s is ${maxMb} MB.`,
    );
  }

  return { ok: true, kind };
}

/**
 * Returns an error message, or null. Unknown durations (the browser cannot decode the
 * format, or reports Infinity for a stream) are not blocked here; the server enforces it.
 */
export function validateVideoDuration(durationSec, limits = DEFAULT_LIMITS) {
  if (!Number.isFinite(durationSec)) return null;
  if (durationSec > limits.max_video_duration_sec) {
    return `This video is ${formatDuration(durationSec)} long. The limit is ${limits.max_video_duration_sec} seconds.`;
  }
  return null;
}