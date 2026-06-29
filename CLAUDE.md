# CLAUDE.md — ml-in-fcs

## Writing style (applies to ALL prose: paper, docs, commit messages, chat)

Goal: prose that reads as written by a careful human researcher, with as little
"AI-generated" feel as possible. Make it sharp, concrete, and specific. Keep the
academic register; do not become informal.

When editing existing prose, make **local edits only**. Do not rewrite
aggressively, do not reorder arguments, and do not change claims, numbers,
citations, labels, or section references unless asked.

### Hard rules

1. **No em-dashes or en-dashes in prose.** Use a period, a comma, parentheses,
   or a colon instead. (LaTeX `--` for numeric ranges is fine; this is about
   sentence punctuation, not math or code.)
2. **Few sentences starting with "The."** A reviewer counted 38+ in a 12-page
   draft and called it extreme. Rework openings (start from the subject, a verb,
   "Because…", "A…"). This includes figure and table captions.
3. **Define every audience-specific term or acronym before first use**, and
   calibrate to the reader. (A reviewer flagged "Jaccard" used five times, never
   defined.) Genuinely basic terms (e.g. MAE) do not need a gloss; use judgment.

### Avoid these constructions

- **"This is X, not Y" / "not X but Y"** and other overly balanced, symmetrical
  sentences that sound too polished.
- **Vague phrases that sound good but carry little information**: "deliberately
  narrow", "creates a sensitive interface", "well motivated", "an appealing
  inductive bias", "usable samples", "a failure mode", "a diagnostic extension".
  Replace each with the concrete fact.
- **Inflated / formulaic openers and transitions**: "We position this work
  as…", "We therefore extend the evaluation with…", "What matters is that…",
  "We note only that…", "Crucially,", "Importantly,", "Under that lens,",
  "A narrower reading fits:".
- **Abstract nouns used to sound technical** without adding precision.
- **Smooth, formulaic colon-explanations** that just restate the clause before.
- **Long parallel lists** of the form "we do X, check Y, and test Z." Break them
  into separate sentences.

### Do instead

- Replace vague abstractions with concrete wording.
- Break up long sentences that carry several claims.
- Keep the tone careful but not over-hedged.
- Prefer the direct statement over the balanced or decorative one.

### Worked example (the target register)

Instead of:

> This separation is useful, but it also creates a sensitive interface:
> downstream results depend on how the learned state chain is initialized,
> reset, and sliced into synthetic training blocks.

write:

> Although useful, this separation makes downstream results sensitive to how the
> learned state chain is initialized, reset, and divided into synthetic training
> blocks.
