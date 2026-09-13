export function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="mb-8 flex flex-col gap-2">
      {eyebrow && (
        <span className="font-mono text-xs uppercase tracking-widest text-[var(--color-gold)]">
          {eyebrow}
        </span>
      )}
      <h1 className="font-serif text-3xl text-[var(--color-text)] sm:text-4xl">
        {title}
      </h1>
      {description && (
        <p className="max-w-2xl text-[var(--color-text-muted)]">{description}</p>
      )}
    </div>
  );
}
