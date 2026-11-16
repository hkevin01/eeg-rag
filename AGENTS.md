# EEG-RAG Agent Guidance

## Repository Orientation

Before broad repository exploration, search `.rtt/context.txt` for relevant paths,
classes, and function signatures, then read the matching source and nearby tests.
The generated index is a structural snapshot; source implementations remain the
authority for behavior and citation correctness. Confirm paths still exist.

After relevant code changes, run `rtt update` from the repository root. Regenerate
the index with the tool rather than editing it manually. Check its generation
header for freshness and run `rtt compare` for current token estimates; counts
quoted in documentation are not current measurements.

See `.rtt/README.md` for tool installation, scope, and local workflow. The index
helps contributors and AI tools navigate; EEG-RAG does not load it at runtime.

## Development Guidance

Follow `.github/copilot-instructions.md` and applicable guidance in `.copilot/`
for Python, EEG terminology, citation verification, and testing requirements.
Keep changes scoped and validate the affected behavior with nearby tests before
regenerating the index. Mock external APIs in unit tests.
