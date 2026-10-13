import { ScanLine } from 'lucide-react';
import { Link, NavLink } from 'react-router-dom';

import ThemeToggle from './ThemeToggle.jsx';

const linkClass = ({ isActive }) =>
  `rounded-lg px-3 py-2 text-sm font-medium transition-colors hover:bg-surface-2 ${
    isActive ? 'text-accent' : 'text-muted'
  }`;

export default function Navbar({ theme, onToggleTheme }) {
  return (
    <header className="sticky top-0 z-40 border-b bg-surface/80 backdrop-blur">
      <nav
        aria-label="Main"
        className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3"
      >
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-accent-fg">
            <ScanLine className="h-5 w-5" aria-hidden="true" />
          </span>
          <span>
            TruthLens <span className="text-accent">AI</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          <NavLink to="/" end className={linkClass}>
            Analyze
          </NavLink>
          <NavLink to="/about" className={linkClass}>
            About
          </NavLink>
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </nav>
    </header>
  );
}