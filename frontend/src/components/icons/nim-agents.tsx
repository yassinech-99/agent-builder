export function NimAgentsLogo({
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
      aria-label="nimAgents logo"
    >
      <rect
        x="3"
        y="3"
        width="42"
        height="42"
        rx="11"
        fill="currentColor"
        fillOpacity="0.12"
        stroke="currentColor"
        strokeWidth="2.4"
      />
      {/* Stylized 'N' tensor path */}
      <path
        d="M14 33V15L25.5 29.5V15"
        stroke="currentColor"
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* NIM compute & routing nodes */}
      <circle cx="33.5" cy="16.5" r="3" fill="currentColor" />
      <circle cx="33.5" cy="31.5" r="3" fill="currentColor" fillOpacity="0.8" />
      <line
        x1="33.5"
        y1="20"
        x2="33.5"
        y2="28"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        opacity="0.65"
      />
    </svg>
  );
}
