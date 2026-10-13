import { Route, Routes } from 'react-router-dom';

import DetectorTierBanner from './components/DetectorTierBanner.jsx';
import DisclaimerNotice from './components/DisclaimerNotice.jsx';
import Navbar from './components/Navbar.jsx';
import { useServerInfo } from './hooks/useServerInfo.js';
import { useTheme } from './hooks/useTheme.js';
import About from './pages/About.jsx';
import Home from './pages/Home.jsx';
import NotFound from './pages/NotFound.jsx';
import Results from './pages/Results.jsx';

export default function App() {
  const { theme, toggle } = useTheme();
  const serverInfo = useServerInfo();

  return (
    <div className="flex min-h-screen flex-col bg-bg text-fg">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-fg"
      >
        Skip to content
      </a>
      <Navbar theme={theme} onToggleTheme={toggle} />
      <DetectorTierBanner status={serverInfo.status} detector={serverInfo.detector} />

      <main id="main" className="flex-1">
        <Routes>
          <Route path="/" element={<Home serverInfo={serverInfo} />} />
          <Route path="/results/:resultId" element={<Results />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <footer className="border-t bg-surface">
        <div className="mx-auto max-w-5xl px-4 py-6">
          <DisclaimerNotice />
          <p className="mt-4 text-xs text-muted">
            TruthLens AI is a student/portfolio project. It is not a forensic tool and must not be
            used as the sole basis for any decision about a person or a piece of media.
          </p>
        </div>
      </footer>
    </div>
  );
}