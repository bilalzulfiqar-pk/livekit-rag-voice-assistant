import clsx from "clsx";

type StatusChipProps = {
  label: string;
  tone: "neutral" | "live" | "success" | "warn" | "accent";
};

export function StatusChip({ label, tone }: StatusChipProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-2 whitespace-nowrap rounded-full border px-3 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.22em] backdrop-blur",
        tone === "neutral" &&
          "border-white/10 bg-white/5 text-[color:var(--color-text-secondary)]",
        tone === "live" &&
          "border-[color:var(--color-cyan)]/35 bg-[color:var(--color-cyan)]/10 text-[color:var(--color-cyan)] shadow-[0_0_24px_rgba(25,211,255,0.18)]",
        tone === "success" &&
          "border-[color:var(--color-accent)]/35 bg-[color:var(--color-accent)]/12 text-[color:var(--color-accent)] shadow-[0_0_24px_rgba(51,240,181,0.16)]",
        tone === "warn" &&
          "border-[color:var(--color-error)]/35 bg-[color:var(--color-error)]/10 text-[color:var(--color-error)]",
        tone === "accent" &&
          "border-[color:var(--color-primary)]/35 bg-[color:var(--color-primary)]/12 text-[color:var(--color-primary)] shadow-[0_0_24px_rgba(124,92,255,0.18)]",
      )}
    >
      <span className="h-2 w-2 shrink-0 rounded-full bg-current" />
      {label}
    </span>
  );
}
