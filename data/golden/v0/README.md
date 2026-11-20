# Golden set v0

30 original (question, context, answer) items for RAG faithfulness: product docs, policies, science,
history and how-tos. All content is invented for this project; no external datasets or personal data.

- `items.json`: the items (`g001`..`g030`, shuffled).
- `labels_template.csv`: the sheet to fill in (`example_id,label,note`).
- `human_labels.csv`: the maintainer's blind labels (round 1, 2026-11-20), notes kept verbatim.
- `intended_labels.json`: the author's intended labels, for later comparison only.

## How to label
Read each item and put `pass` or `fail` in the `label` column. **pass** = every claim in the answer is
supported by the context. **fail** = at least one claim is not supported (wrong, extra or overstated).
Judge from the context only, not from what you know. Use `note` for anything borderline.

**WARNING: do NOT open `intended_labels.json` before you have finished labelling.** It would bias the
labels and invalidate the agreement numbers.
