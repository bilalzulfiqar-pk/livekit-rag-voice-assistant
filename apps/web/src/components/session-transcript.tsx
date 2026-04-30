import { useEffect, useRef } from "react";

import clsx from "clsx";

import type { TranscriptEntry } from "@/lib/voice";

type SessionTranscriptProps = {
  entries: TranscriptEntry[];
  emptyMessage?: string;
};

export function SessionTranscript({
  entries,
  emptyMessage = "Start a conversation and your transcript will appear here in real time.",
}: SessionTranscriptProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    container.scrollTop = container.scrollHeight;
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="rounded-[1.6rem] border border-dashed border-white/10 bg-white/[0.03] p-6 text-sm leading-6 text-[color:var(--color-text-secondary)]">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="auralis-scrollbar flex max-h-[30rem] flex-col gap-3 overflow-y-auto pr-1"
    >
      {entries.map((entry) => (
        <article
          key={entry.id}
          className={clsx(
            "max-w-[88%] rounded-[1.4rem] px-4 py-3 shadow-[0_18px_40px_rgba(3,8,17,0.24)]",
            entry.role === "assistant"
              ? "self-start border border-white/10 bg-white/[0.05] text-[color:var(--color-text-primary)]"
              : "self-end border border-[color:var(--color-primary)]/30 bg-[linear-gradient(135deg,rgba(124,92,255,0.32),rgba(25,211,255,0.18))] text-[color:var(--color-text-primary)]",
            !entry.isFinal && "opacity-60",
            entry.isInterrupted &&
              "border-[color:var(--color-warning)]/40 shadow-[0_0_0_1px_rgba(255,184,77,0.18)]",
          )}
        >
          <div className="mb-2 flex items-center gap-2 text-[0.68rem] font-semibold uppercase tracking-[0.18em]">
            <span className="opacity-75">
              {entry.role === "assistant" ? "Assistant" : "You"}
            </span>
            {!entry.isFinal && (
              <span className="rounded-full bg-white/10 px-2 py-0.5 text-[0.58rem] text-[color:var(--color-text-secondary)]">
                Partial
              </span>
            )}
            {entry.isInterrupted && (
              <span className="rounded-full bg-[color:var(--color-warning)]/15 px-2 py-0.5 text-[0.58rem] text-[color:var(--color-warning)]">
                Interrupted
              </span>
            )}
          </div>
          <p className="text-sm leading-6">
            {entry.text}
            {entry.isInterrupted ? "..." : ""}
          </p>
        </article>
      ))}
    </div>
  );
}
