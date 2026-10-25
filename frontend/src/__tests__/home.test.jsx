import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import Home from '../pages/Home.jsx';
import { ApiError, detectImage } from '../services/api.js';
import { DEFAULT_LIMITS } from '../utils/validation.js';

vi.mock('../services/api.js', async (importOriginal) => {
  const actual = await importOriginal();
  return { ...actual, detectImage: vi.fn(), detectVideo: vi.fn() };
});

const MB = 1024 * 1024;

function renderHome() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route
          path="/"
          element={<Home serverInfo={{ limits: DEFAULT_LIMITS, limitsFromServer: true }} />}
        />
        <Route path="/results/:resultId" element={<p>RESULT PAGE</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

const fileInput = () => document.querySelector('input[type="file"]');
const png = () => new File(['x'], 'photo.png', { type: 'image/png' });

beforeEach(() => {
  detectImage.mockReset();
  URL.createObjectURL = vi.fn(() => 'blob:mock'); // jsdom has no object URLs
  URL.revokeObjectURL = vi.fn();
});

describe('Home upload flow', () => {
  it('rejects an unsupported file type before uploading anything', async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderHome();
    await user.upload(fileInput(), new File(['x'], 'notes.pdf', { type: 'application/pdf' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/unsupported file type/i);
    expect(detectImage).not.toHaveBeenCalled();
  });

  it('rejects an oversized image with the limit in the message', async () => {
    const user = userEvent.setup({ applyAccept: false });
    renderHome();
    const big = png();
    Object.defineProperty(big, 'size', { value: 11 * MB });
    await user.upload(fileInput(), big);
    expect(await screen.findByRole('alert')).toHaveTextContent(/limit for images is 10 MB/i);
  });

  it('rejects more than one dropped file', async () => {
    renderHome();
    const zone = screen.getByText('Drag and drop an image or video here').closest('div');
    fireEvent.drop(zone, { dataTransfer: { files: [png(), png()], types: ['Files'] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Please choose one file at a time.');
  });

  it('shows the selected file with an Analyze button', async () => {
    renderHome();
    await userEvent.upload(fileInput(), png());
    expect(await screen.findByText('Selected image')).toBeInTheDocument();
    expect(screen.getByText('photo.png')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Media' })).toBeEnabled();
  });

  it('uploads the file and moves to the results page', async () => {
    detectImage.mockResolvedValue({ id: 'a'.repeat(32) });
    renderHome();
    const file = png();
    await userEvent.upload(fileInput(), file);
    await userEvent.click(await screen.findByRole('button', { name: 'Analyze Media' }));
    expect(await screen.findByText('RESULT PAGE')).toBeInTheDocument();
    expect(detectImage).toHaveBeenCalledWith(file, expect.any(Object));
  });

  it('offers a retry for a transient server error and shows the reference', async () => {
    detectImage.mockRejectedValue(
      new ApiError('Too many requests. Try again in 7 s.', {
        code: 'RATE_LIMITED', status: 429, requestId: 'req-1', retryAfter: 7,
      }),
    );
    renderHome();
    await userEvent.upload(fileInput(), png());
    await userEvent.click(await screen.findByRole('button', { name: 'Analyze Media' }));
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('Too many requests');
    expect(alert).toHaveTextContent('Reference: req-1');
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument();
  });

  it('offers no retry when resending the same file cannot succeed', async () => {
    detectImage.mockRejectedValue(
      new ApiError('File content does not match a supported image format.', {
        code: 'UNSUPPORTED_MEDIA_TYPE', status: 415, requestId: 'req-2',
      }),
    );
    renderHome();
    await userEvent.upload(fileInput(), png());
    await userEvent.click(await screen.findByRole('button', { name: 'Analyze Media' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('does not match');
    expect(screen.queryByRole('button', { name: 'Try again' })).not.toBeInTheDocument();
  });
});