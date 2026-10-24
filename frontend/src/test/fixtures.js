export const imageResult = {
  id: 'a'.repeat(32),
  media_type: 'image',
  classification: 'UNCERTAIN',
  classification_label: 'Uncertain',
  confidence: 0.58,
  real_probability: 0.58,
  ai_probability: 0.42,
  risk_level: 'MEDIUM',
  processing_time_ms: 94,
  model: 'demo-heuristic',
  detector_version: '0.1.0',
  detector_tier: 'DEMO_HEURISTIC',
  tier_notice: 'Prototype mode: this result comes from simple heuristics, not a trained AI detector.',
  score_is_calibrated: false,
  uncertain_band: [0.35, 0.65],
  explanation: [
    {
      key: 'high_frequency_energy',
      title: 'High-frequency spectral energy',
      detail: 'Share of spectral energy in the upper half of the frequency range.',
      source: 'heuristic',
      scored: true,
      value: -2.31,
      unit: 'log10 ratio',
      direction: 'toward_synthetic',
    },
    {
      key: 'exif_presence',
      title: 'EXIF metadata',
      detail: 'No EXIF metadata found. This is NOT evidence either way.',
      source: 'forensic',
      scored: false,
      value: null,
      unit: null,
      direction: 'neutral',
    },
  ],
  explanation_notice: 'These items are measured image statistics.',
  heatmap_status: 'unsupported',
  heatmap_notice:
    'The prototype detector measures whole-image statistics and has no regions to highlight, so no heatmap is shown.',
  heatmap: null,
  metadata: {
    width: 64, height: 64, megapixels: 0, format: 'png', color_mode: 'RGB',
    has_alpha: false, has_exif: false, size_bytes: 1234,
  },
  created_at: '2026-10-12T10:15:30Z',
  disclaimer:
    'AI detection is probabilistic. Results should not be treated as definitive proof of authenticity or manipulation.',
};

const frame = (number, seconds, probability) => ({
  frame_number: number,
  timestamp_sec: seconds,
  ai_probability: probability,
  selection: 'uniform',
  flagged: probability >= 0.65,
});

export const videoResult = {
  ...imageResult,
  id: 'b'.repeat(32),
  media_type: 'video',
  frames_analyzed: 3,
  frame_results: [frame(5, 0.5, 0.4), frame(15, 1.5, 0.5), frame(25, 2.5, 0.6)],
  highest_scoring_frames: [frame(25, 2.5, 0.6), frame(15, 1.5, 0.5), frame(5, 0.5, 0.4)],
  sampling: {
    strategy: 'uniform positions plus strongest scene-change candidates',
    max_frames_requested: 24, frames_analyzed: 3, uniform_frames: 3, scene_change_frames: 0,
    frames_decoded: 30, coverage_fraction: 0.1, description: 'Frames between samples are not analyzed.',
  },
  aggregation: {
    method: '0.70 x trimmed mean + 0.30 x mean of top 25% of frames',
    rule_note: 'A design choice. Not fitted to data.',
    flag_threshold: 0.65, score: 0.5, mean: 0.5, trimmed_mean: 0.5, top_quarter_mean: 0.6,
    std_dev: 0.08, minimum: 0.4, maximum: 0.6, flagged_fraction: 0,
  },
  warnings: [
    'Frames are scored independently. Frame-to-frame (temporal) artifacts, audio and lip-sync are not examined.',
  ],
  metadata: {
    width: 64, height: 64, megapixels: 0, fps: 10, duration_sec: 3,
    frame_count_reported: 30, frames_decoded: 30, codec: 'MJPG', size_bytes: 1000,
  },
};