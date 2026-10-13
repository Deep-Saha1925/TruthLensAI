import { Moon, Sun } from 'lucide-react';

export default function ThemeToggle({ theme, onToggle }) {
  const next = theme === 'dark' ? 'light' : 'dark';
  const Icon = theme === 'dark' ? Sun : Moon;

  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      className="inline-flex h-10 w-10 items-center justify-center rounded-lg border bg-surface text-fg transition-colors hover:bg-surface-2"
    >
      <Icon className="h-5 w-5" aria-hidden="true" />
    </button>
  );
}