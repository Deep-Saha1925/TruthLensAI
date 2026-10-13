import { UploadCloud } from 'lucide-react';
import { useId, useRef, useState } from 'react';

import { formatDuration } from '../utils/formatting.js';

const hasFiles = (event) => Array.from(event.dataTransfer?.types ?? []).includes('Files');

/**
 * Drag-and-drop + click-to-browse picker. It only reports the chosen files;
 * validation and state live in the parent.
 */
export default function UploadZone({ limits, onFilesSelected, disabled = false }) {
  const inputRef = useRef(null);
  const dragDepth = useRef(0); // dragenter/leave also fire for child elements
  const [dragging, setDragging] = useState(false);
  const hintId = useId();

  const accept = [...limits.image_extensions, ...limits.video_extensions].join(',');

  const openPicker = () => {
    if (!disabled) inputRef.current?.click();
  };

  const emit = (list) => {
    const files = Array.from(list ?? []);
    if (files.length > 0) onFilesSelected(files);
  };

  const handleDragEnter = (event) => {
    if (disabled || !hasFiles(event)) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };

  const handleDragOver = (event) => {
    if (disabled || !hasFiles(event)) return;
    event.preventDefault(); // required, or the browser will not allow a drop
  };

  const handleDragLeave = () => {
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (!disabled) emit(event.dataTransfer.files);
  };

  return (
    <>
      <div
        onClick={openPicker}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-colors ${
          dragging ? 'border-accent bg-accent/10' : 'bg-surface hover:bg-surface-2'
        } ${disabled ? 'cursor-not-allowed opacity-60' : ''}`}
      >
        <UploadCloud className="h-10 w-10 text-accent" aria-hidden="true" />
        <p className="text-lg font-semibold">
          {dragging ? 'Drop the file to select it' : 'Drag and drop an image or video here'}
        </p>
        <p className="text-sm text-muted">or</p>
        {/* Enter/Space on this button fires a click that bubbles to the zone's handler. */}
        <button
          type="button"
          disabled={disabled}
          aria-describedby={hintId}
          className="rounded-lg bg-accent px-5 py-2.5 font-medium text-accent-fg transition-opacity hover:opacity-90 disabled:cursor-not-allowed"
        >
          Choose a file
        </button>
        <p id={hintId} className="max-w-md text-xs text-muted">
          Images: {limits.image_extensions.join(' ')} up to {limits.max_image_size_mb} MB. Videos:{' '}
          {limits.video_extensions.join(' ')} up to {limits.max_video_size_mb} MB and{' '}
          {formatDuration(limits.max_video_duration_sec)} minutes.
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onClick={(event) => event.stopPropagation()} // do not re-trigger the zone's click handler
        onChange={(event) => {
          emit(event.target.files);
          event.target.value = ''; // allow choosing the same file again
        }}
      />
    </>
  );
}