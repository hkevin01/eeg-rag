# rtt: Reduce the Tokens

rtt (the `reducethemtokens` CLI package) compresses a repository into a structural
source index. EEG-RAG stores that generated artifact at `.rtt/context.txt` so
contributors and AI tools can locate relevant code before broad exploration.

## Repository Integration

| Item | Repository location | Role | Generated? |
| --- | --- | --- | --- |
| Structural index | `.rtt/context.txt` | Compact file and symbol map | Yes |
| Agent guidance | `AGENTS.md` | Workflow and index pointer | No |
| Copilot guidance | `.github/copilot-instructions.md` | Agent orientation | No |
| Tool guide | `.rtt/README.md` | Local workflow and limitations | No |
| Application source | `src/eeg_rag/` | EEG-RAG implementation | No |
| Tests | `tests/` | Behavioral verification | No |
| Supporting code | `scripts/`, `examples/` | Operations and demonstrations | No |

The index and its consumer instructions are separate artifacts. EEG-RAG's API,
Streamlit UI, storage, retrieval, and tests work independently of rtt. It is a
contributor tool, not an application dependency or runtime retrieval corpus.

## Local Workflow

The CLI is already installed in the environment used to create this integration.
The installed version is `reducethemtokens 0.6.1`, which requires Python 3.10+
independently of EEG-RAG's Python 3.9+ support. For a new contributor environment,
install it separately from EEG-RAG:

```sh
pipx install reducethemtokens
rtt --help
```

Run commands from the repository root:

```sh
# Generate or refresh the committed structural snapshot.
rtt update

# Refresh and show structural changes since the previous snapshot.
rtt update --diff

# Measure current source-to-skeleton token estimates.
rtt compare
```

Regenerate after changes to indexed files, paths, imports, classes, or signatures.
Use the generator rather than manually editing `.rtt/context.txt`. Review the
generated diff with your source changes. The `.rtt-cache/` parsing cache is ignored
by Git; `.rtt/context.txt` remains eligible for version control.

Default indexing uses the installed tool's supported parsers and file-selection
rules. Keep tests in the index so agents can find behavioral checks. The installed
CLI also supports repeatable `--include` and `--exclude` globs, `--no-tests`, and
`--max-tokens`; consult `rtt update --help` before changing scope. If you use custom
filters, repeat them on later updates. A filtered or trimmed artifact is not
directly comparable to a default-scope `rtt compare` result.

## Consulting the Index

Search for the task's concept, symbol, or path rather than loading the entire
artifact into every prompt. For example:

```sh
rg -n 'CitationVerifier|citation_verifier|HybridRetriever' .rtt/context.txt
```

Use matches to select the owning source file and nearby tests. Read the actual
implementation before making behavioral claims or edits. Verify file existence
and use source searches when the index has no match or may be stale.

## What the Index Contains

Exact output depends on the installed rtt version, included files, and parsers.
The generation header reports snapshot metadata, indexed file count, and a
skeleton token estimate. Typical file entries include these fields:

| Field | Navigation use | Limitation |
| --- | --- | --- |
| Path and language | Select a file and interpret its syntax | Does not prove freshness or compiler settings |
| Imports | Locate dependencies and likely layers | Does not establish data flow |
| Type names | Find an owning abstraction | Does not describe invariants or inheritance |
| Functions and signatures | Locate likely entry points and API shapes | Does not reveal branches, returns, or side effects |
| Member indentation | Identify containment within a type | Does not establish runtime semantics |
| Generation header | Assess snapshot freshness and size | Does not measure exact task savings |

An import can suggest dependency direction, a type can suggest ownership, and a
signature can narrow a source read. None substitutes for executable source,
tests, or verified PubMed citations. Parser omissions and token-budget trimming
can leave declarations absent from the snapshot.

## Freshness and Statistics

Read the current index header and run the installed `rtt compare` command for
current statistics. Counts copied into guidance may drift and should be treated
as historical snapshots, not live measurements. Token estimates describe the
tool's representation, not guaranteed savings for a particular model or task.

If rtt is unavailable, consult source directly and install the contributor tool
before the next regeneration; application execution does not require it.

## Upstream Reference

Tool documentation and source: <https://github.com/yttrium400/reducethemtokens>.
Use the installed CLI's `--help` output when upstream documentation differs from
your installed version. This integration uses manual regeneration and does not
install git hooks or automatic commits.
