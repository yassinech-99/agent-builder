export function NovaGridLogo({
  className,
  width = 32,
  height = 32,
}: {
  className?: string;
  width?: number;
  height?: number;
}) {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Grid pattern */}
      <rect x="4" y="4" width="12" height="12" rx="3" fill="currentColor" opacity="0.9" />
      <rect x="18" y="4" width="12" height="12" rx="3" fill="currentColor" opacity="0.6" />
      <rect x="32" y="4" width="12" height="12" rx="3" fill="currentColor" opacity="0.3" />
      <rect x="4" y="18" width="12" height="12" rx="3" fill="currentColor" opacity="0.6" />
      <rect x="18" y="18" width="12" height="12" rx="3" fill="currentColor" opacity="1" />
      <rect x="32" y="18" width="12" height="12" rx="3" fill="currentColor" opacity="0.6" />
      <rect x="4" y="32" width="12" height="12" rx="3" fill="currentColor" opacity="0.3" />
      <rect x="18" y="32" width="12" height="12" rx="3" fill="currentColor" opacity="0.6" />
      <rect x="32" y="32" width="12" height="12" rx="3" fill="currentColor" opacity="0.9" />
      {/* Connection lines */}
      <line x1="10" y1="16" x2="10" y2="18" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="24" y1="16" x2="24" y2="18" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="38" y1="16" x2="38" y2="18" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="10" y1="30" x2="10" y2="32" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="24" y1="30" x2="24" y2="32" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="38" y1="30" x2="38" y2="32" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="10" x2="18" y2="10" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="30" y1="10" x2="32" y2="10" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="24" x2="18" y2="24" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="30" y1="24" x2="32" y2="24" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="38" x2="18" y2="38" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
      <line x1="30" y1="38" x2="32" y2="38" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
    </svg>
  );
}
