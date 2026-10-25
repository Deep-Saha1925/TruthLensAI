import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import ClassificationBadge from '../components/ClassificationBadge.jsx';
import ConfidenceMeter from '../components/ConfidenceMeter.jsx';
import DetectorTierBanner from '../components/DetectorTierBanner.jsx';
import ErrorMessage from '../components/ErrorMessage.jsx';
import ExplainabilityPanel from '../components/ExplainabilityPanel.jsx';
import ModelPerformancePanel from '../components/ModelPerformancePanel.jsx';
import ProbabilityChart from '../components/ProbabilityChart.jsx';
import UploadZone from '../components/UploadZone.jsx';
import { imageResult } from '../test/fixtures.js';
import { DEFAULT_LIMITS } from '../utils/validation.js';

describe('ClassificationBadge', () => {
  it('shows the label as text with the tone of the classification', () => {
    render(<ClassificationBadge classification="LIKELY_SYNTHETIC" label="Likely AI-generated or manipulated" />);
    const badge = screen.getByText('Likely AI-generated or manipulated');
    expect(badge).toBeInTheDocument();
    expect(badge.className).toContain('text-danger-fg');
  });
});

describe('ConfidenceMeter', () => {
  it('exposes the value and says it is not a measure of correctness', () => {
    render(<ConfidenceMeter confidence={0.58} aiProbability={0.42} calibrated={false} />);
    const meter = screen.getByRole('meter', { name: /confidence/i });
    expect(meter).toHaveAttribute('aria-valuenow', '58');
    expect(meter).toHaveAttribute('aria-valuetext', '58% toward Real');
    expect(screen.getByText(/not a calibrated probability/i)).toBeInTheDocument();
  });
});

describe('ProbabilityChart', () => {
  it('labels both bars numerically and names the zone the score falls in', () => {
    render(<ProbabilityChart aiProbability={0.4188} uncertainBand={[0.35, 0.65]} calibrated={false} />);
    expect(screen.getByRole('meter', { name: 'AI-generated or manipulated' })).toHaveAttribute('aria-valuenow', '42');
    expect(screen.getByRole('meter', { name: 'Real' })).toHaveAttribute('aria-valuenow', '58');
    expect(screen.getByText('Uncertain', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText(/not calibrated probabilities/i)).toBeInTheDocument();
  });
});

describe('ErrorMessage', () => {
  it('announces the error, shows the reference and offers a retry', async () => {
    const onRetry = vi.fn();
    render(<ErrorMessage message="Too many requests." requestId="req-1" onRetry={onRetry} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Too many requests.');
    expect(screen.getByText('Reference: req-1')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('offers no retry unless asked', () => {
    render(<ErrorMessage message="Bad file." />);
    expect(screen.queryByRole('button', { name: 'Try again' })).not.toBeInTheDocument();
  });
});

describe('DetectorTierBanner', () => {
  it('reports an unreachable server as an alert', () => {
    render(<DetectorTierBanner status="unreachable" detector={null} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Server unreachable');
  });

  it('reports a server whose model did not load', () => {
    render(<DetectorTierBanner status="ready" detector={null} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Model not loaded');
  });

  it('labels the demo detector as prototype mode', () => {
    render(<DetectorTierBanner status="ready" detector={{ tier: 'DEMO_HEURISTIC', tier_notice: 'Prototype notice' }} />);
    const banner = screen.getByRole('status');
    expect(banner).toHaveTextContent('Prototype mode');
    expect(banner).toHaveTextContent('Prototype notice');
  });
});

describe('ExplainabilityPanel', () => {
  const base = {
    explanation: imageResult.explanation,
    notice: 'Notice',
    tier: 'DEMO_HEURISTIC',
    mediaType: 'image',
    originalFile: null,
  };

  it('separates items that fed the score from context-only items', () => {
    render(<ExplainabilityPanel {...base} heatmap={null} heatmapNotice="No heatmap here." />);
    expect(screen.getByRole('heading', { name: /items that fed the score/i })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /context only/i })).toBeInTheDocument();
    expect(screen.getByText('Leans toward synthetic')).toBeInTheDocument();
  });

  it('explains why there is no heatmap instead of drawing one', () => {
    render(<ExplainabilityPanel {...base} heatmap={null} heatmapNotice="No heatmap here." />);
    expect(screen.getByText('No heatmap here.')).toBeInTheDocument();
    expect(screen.queryByAltText(/heatmap of the regions/i)).not.toBeInTheDocument();
  });

  it('shows the server heatmap and lets the user hide it', async () => {
    const heatmap = {
      method: 'grad-cam', target: 'synthetic-vs-real log-odds', width: 8, height: 8,
      data_url: 'data:image/png;base64,AAAA',
    };
    render(<ExplainabilityPanel {...base} tier="TRAINED" heatmap={heatmap} heatmapNotice="Heatmap notice" />);
    expect(screen.getByAltText(/heatmap of the regions/i)).toHaveAttribute('src', heatmap.data_url);

    const slider = screen.getByRole('slider', { name: /opacity/i });
    expect(slider).toBeEnabled();
    await userEvent.click(screen.getByRole('checkbox', { name: /show heatmap/i }));
    expect(slider).toBeDisabled();
  });
});

describe('ModelPerformancePanel', () => {
  const detector = {
    name: 'demo-heuristic', version: '0.1.0', tier: 'TRAINED',
    evaluation: { accuracy: 0.9, roc_auc: 0.95, test_images: 480, warning: 'Evaluated on only 480 test images.' },
  };

  it('shows measured metrics and any warning for the matching model', () => {
    render(<ModelPerformancePanel detector={detector} result={imageResult} />);
    expect(screen.getByRole('heading', { name: /measured performance/i })).toBeInTheDocument();
    expect(screen.getByText('90.0%')).toBeInTheDocument();
    expect(screen.getByText('Evaluated on only 480 test images.')).toBeInTheDocument();
  });

  it('shows nothing when the server runs a different model version', () => {
    const { container } = render(
      <ModelPerformancePanel detector={{ ...detector, version: '9.9' }} result={imageResult} />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

describe('UploadZone', () => {
  const file = new File(['x'], 'notes.pdf', { type: 'application/pdf' });

  it('reports a chosen file and leaves validation to the parent', async () => {
    const user = userEvent.setup({ applyAccept: false });
    const onFilesSelected = vi.fn();
    const { container } = render(<UploadZone limits={DEFAULT_LIMITS} onFilesSelected={onFilesSelected} />);
    await user.upload(container.querySelector('input[type="file"]'), file);
    expect(onFilesSelected).toHaveBeenCalledWith([file]);
  });

  it('accepts dropped files', () => {
    const onFilesSelected = vi.fn();
    render(<UploadZone limits={DEFAULT_LIMITS} onFilesSelected={onFilesSelected} />);
    const zone = screen.getByText('Drag and drop an image or video here').closest('div');
    fireEvent.drop(zone, { dataTransfer: { files: [file], types: ['Files'] } });
    expect(onFilesSelected).toHaveBeenCalledWith([file]);
  });

  it('ignores drops while disabled and advertises the supported types', () => {
    const onFilesSelected = vi.fn();
    const { container } = render(
      <UploadZone limits={DEFAULT_LIMITS} onFilesSelected={onFilesSelected} disabled />,
    );
    const zone = screen.getByText('Drag and drop an image or video here').closest('div');
    fireEvent.drop(zone, { dataTransfer: { files: [file], types: ['Files'] } });
    expect(onFilesSelected).not.toHaveBeenCalled();
    expect(container.querySelector('input[type="file"]').getAttribute('accept')).toContain('.png');
  });
});