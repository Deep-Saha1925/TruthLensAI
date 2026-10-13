import DisclaimerNotice from '../components/DisclaimerNotice.jsx';

const TIERS = [
  {
    name: 'Prototype mode (demo heuristic)',
    text: 'Hand-written signal measurements (spectral energy, noise strength) mapped to a score with placeholder thresholds. Not fitted to any dataset and with no demonstrated accuracy. It exists so the application runs end to end.',
  },
  {
    name: 'Third-party pretrained model',
    text: 'An off-the-shelf classifier that has not been validated on this project\'s data. Its accuracy on new generators and on compressed or resized images is unknown.',
  },
  {
    name: 'Trained model',
    text: 'A model trained and evaluated with this project\'s pipeline, shipped with a report of measured metrics (accuracy, precision, recall, F1, ROC-AUC) and known weaknesses.',
  },
];

const LIMITATIONS = [
  ['False positives', 'Real photos can be flagged: heavy denoising, beauty filters, screenshots and strong compression all resemble synthetic statistics.'],
  ['False negatives', 'Synthetic media can pass as real, especially from generators the detector has never seen.'],
  ['Dataset bias', 'A detector inherits the biases of its training data: subjects, cameras, image styles and generators that were over- or under-represented.'],
  ['Generalization', 'Performance measured on one set of generators usually drops on others. Test-set accuracy does not guarantee real-world accuracy.'],
  ['New generative models', 'Detectors age quickly. Each new generator can introduce different artifacts or remove old ones.'],
  ['Compression and resizing', 'Re-saving, scaling and cropping destroy the fine-grained traces many detectors rely on.'],
  ['Social-media recompression', 'Platforms re-encode uploads, which can erase evidence of synthesis and add artifacts to real media.'],
  ['Adversarial manipulation', 'A determined person can deliberately alter content to evade detection.'],
  ['Uncertainty', 'Scores are estimates. Unless a model is calibrated, a score of 0.8 does not mean "80% likely". Scores near the middle are reported as "uncertain".'],
];

export default function About() {
  return (
    <div className="mx-auto max-w-3xl space-y-12 px-4 py-12">
      <header>
        <h1 className="text-3xl font-bold">About TruthLens AI</h1>
        <p className="mt-3 text-muted">
          TruthLens AI estimates whether an image or video shows signs of being AI-generated or
          manipulated. It reports probabilities and uncertainty, never a verdict.
        </p>
        <DisclaimerNotice className="mt-4" />
      </header>

      <section aria-labelledby="tiers-title">
        <h2 id="tiers-title" className="text-xl font-bold">
          What is behind a result
        </h2>
        <p className="mt-2 text-muted">
          Every result is labelled with the kind of detector that produced it, so you always know
          how much weight it deserves.
        </p>
        <ul className="mt-4 space-y-3">
          {TIERS.map((tier) => (
            <li key={tier.name} className="rounded-xl border bg-surface p-4">
              <h3 className="font-semibold">{tier.name}</h3>
              <p className="mt-1 text-sm text-muted">{tier.text}</p>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="video-title">
        <h2 id="video-title" className="text-xl font-bold">
          How videos are analyzed
        </h2>
        <p className="mt-2 text-muted">
          A limited number of frames are sampled (evenly spaced, plus some at detected scene
          changes), each frame is scored on its own, and the scores are combined so that a
          concentrated run of high-scoring frames can lift the result. Frames are not compared
          with one another, and audio and lip-sync are not examined. A brief manipulated segment
          between sampled frames can be missed.
        </p>
      </section>

      <section aria-labelledby="limits-title">
        <h2 id="limits-title" className="text-xl font-bold">
          Limitations
        </h2>
        <dl className="mt-4 space-y-3">
          {LIMITATIONS.map(([term, text]) => (
            <div key={term} className="rounded-xl border bg-surface p-4">
              <dt className="font-semibold">{term}</dt>
              <dd className="mt-1 text-sm text-muted">{text}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section aria-labelledby="privacy-title">
        <h2 id="privacy-title" className="text-xl font-bold">
          Privacy
        </h2>
        <p className="mt-2 text-muted">
          Uploaded files are processed in a temporary location and deleted immediately after
          analysis. Only the scores and technical details are kept, in server memory, for a short
          time (15 minutes by default) so a result page can be reopened.
        </p>
      </section>
    </div>
  );
}