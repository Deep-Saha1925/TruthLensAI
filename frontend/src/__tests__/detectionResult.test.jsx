import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import DetectionResult from '../components/DetectionResult.jsx';
import { imageResult, videoResult } from '../test/fixtures.js';

// jsdom has no layout, so the chart library renders nothing useful: stub it.
vi.mock('recharts', () => {
  const Pass = ({ children }) => children ?? null;
  const Nothing = () => null;
  return {
    ResponsiveContainer: Pass, LineChart: Pass, CartesianGrid: Nothing, Line: Nothing,
    ReferenceArea: Nothing, ReferenceLine: Nothing, Tooltip: Nothing, XAxis: Nothing, YAxis: Nothing,
  };
});

const renderResult = (result, props = {}) =>
  render(
    <MemoryRouter>
      <DetectionResult result={result} media={null} detector={null} {...props} />
    </MemoryRouter>,
  );

// The collapsed "Raw response (JSON)" block repeats every value on the page, so text
// queries must skip it.
const byText = (matcher) => screen.getByText(matcher, { ignore: 'script, style, pre' });

describe('DetectionResult (image)', () => {
  it('moves focus to the heading and shows the classification and tier notice', () => {
    renderResult(imageResult);
    expect(screen.getByRole('heading', { name: 'Analysis result' })).toHaveFocus();

    const summary = screen.getByRole('region', { name: 'Summary' });
    expect(within(summary).getByText('Uncertain')).toBeInTheDocument();
    expect(within(summary).getByText(/raw score, not a calibrated probability/i)).toBeInTheDocument();

    expect(byText(/this result comes from simple heuristics/i)).toBeInTheDocument();
  });

  it('explains that the original is not available instead of showing nothing', () => {
    renderResult(imageResult);
    expect(byText(/The original is not shown/i)).toBeInTheDocument();
  });

  it('explains why there is no heatmap', () => {
    renderResult(imageResult);
    expect(byText(/has no regions to highlight/i)).toBeInTheDocument();
  });

  it('keeps the disclaimer and a way to start over', () => {
    renderResult(imageResult);
    expect(screen.getAllByText(/AI detection is probabilistic/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('link', { name: 'Analyze Another File' })).toHaveAttribute('href', '/');
  });

  it('shows measured performance only for the model that produced the result', () => {
    const detector = {
      name: 'demo-heuristic', version: '0.1.0', tier: 'TRAINED',
      evaluation: { accuracy: 0.9, roc_auc: 0.95, test_images: 480 },
    };
    renderResult(imageResult, { detector });
    expect(screen.getByRole('heading', { name: /measured performance/i })).toBeInTheDocument();
  });

  it('hides the performance panel when the server runs a different model', () => {
    const detector = {
      name: 'another-model', version: '1', tier: 'TRAINED',
      evaluation: { accuracy: 0.9 },
    };
    renderResult(imageResult, { detector });
    expect(screen.queryByRole('heading', { name: /measured performance/i })).not.toBeInTheDocument();
  });
});

describe('DetectionResult (video)', () => {
  it('shows the timeline, a table of frame scores and the warnings', () => {
    renderResult(videoResult);
    expect(screen.getByRole('heading', { name: 'Detection timeline' })).toBeInTheDocument();
    expect(byText(/Read before relying on this video result/)).toBeInTheDocument();
    expect(byText(/Frames are scored independently/)).toBeInTheDocument();

    // The table sits in a closed <details>, so include inaccessible elements in the query.
    const table = screen.getByRole('table', { name: /AI score for each analyzed frame/i, hidden: true });
    expect(within(table).getAllByRole('row', { hidden: true })).toHaveLength(
      1 + videoResult.frame_results.length,
    );
  });

  it('lists the highest-scoring frames but cannot seek without the original video', () => {
    renderResult(videoResult);
    expect(screen.getByRole('heading', { name: 'Highest-scoring frames' })).toBeInTheDocument();
    const buttons = screen.getAllByRole('button', { name: 'Jump to this moment' });
    expect(buttons).toHaveLength(videoResult.highest_scoring_frames.length);
    buttons.forEach((button) => expect(button).toBeDisabled());
  });

  it('explains how the video score was built', () => {
    renderResult(videoResult);
    expect(screen.getByRole('heading', { name: 'How the video score was built' })).toBeInTheDocument();
    expect(byText(/0\.70 x trimmed mean/)).toBeInTheDocument();
  });
});