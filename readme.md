# Secure Enterprise RAG

A small Python command-line prototype for retrieval-augmented generation (RAG) with role-based access control (RBAC). It indexes local text documents, retrieves relevant passages for a user's question, filters those passages by the user's role, and can optionally send only the authorized context to an OpenAI model.

> **Status:** Learning and demonstration project. Authentication, storage, retrieval diagnostics, and logging are intentionally simple and are not production security controls.

## How it works

1. Read the sample documents from `docs/`.
2. Split each document into overlapping character-based chunks (300 characters, with 50 characters of overlap by default).
3. Embed the chunks using the `all-MiniLM-L6-v2` Sentence Transformers model and add them to an in-memory FAISS index.
4. Prompt for a username and map that username to a role.
5. Embed each question and search for up to 10 nearest chunks.
6. Keep only chunks allowed for the user's role, then discard results whose L2 distance is at or above the configured relevance threshold.
7. Print the accessible context. If LLM mode is enabled, build a prompt from that context and request an answer from OpenAI.
8. Append the query and retrieval details to `logs.jsonl` when accessible results are found.

The index is rebuilt from the files each time the program starts; it is not persisted between runs.

## Project layout

```text
secure-enterprise-rag/
├── docs/
│   ├── doc1.txt          # Finance example; manager and admin access
│   └── doc2.txt          # Engineering example; all configured roles
├── main.py               # Indexing, retrieval, access checks, and CLI loop
├── requirements.txt      # Python dependencies
├── blueprint.md          # Longer-term architecture and development roadmap
└── readme.md
```

## Requirements

- Python 3.10 or later recommended
- Dependencies listed in `requirements.txt`
- An OpenAI API key only if you enable LLM mode

## Setup

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

The first run downloads the `all-MiniLM-L6-v2` embedding model if it is not already available locally. FAISS and Sentence Transformers may require platform-compatible Python wheels.

## Run

```bash
python main.py
```

Enter one of the usernames configured in `main.py`:

| Username | Role |
| --- | --- |
| `ujjwal` | `employee` |
| `kamal` | `manager` |
| `admin` | `admin` |

Type a question at the `>>` prompt. The program continues accepting questions until you interrupt it (for example, with `Ctrl+C`).

The bundled example documents demonstrate the access rules:

- `docs/doc1.txt` contains finance information and is allowed for `manager` and `admin`.
- `docs/doc2.txt` contains engineering information and is allowed for `employee`, `manager`, and `admin`.

## Enable generated answers (optional)

LLM calls are disabled by default (`USE_LLM = False` in `main.py`). To enable them, set `USE_LLM = True` and configure `OPENAI_API_KEY` in your environment before starting the program.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python main.py
```

macOS/Linux:

```bash
export OPENAI_API_KEY="your-api-key"
python main.py
```

When enabled, the configured OpenAI client calls `gpt-4.1-mini`. The prompt instructs the model to answer from the supplied context and say “I don't know” when the answer is not present.

## Configuration

The main settings are defined near the top of `main.py`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `DOCS_PATH` | `docs` | Declared document directory (the current loader uses explicit file paths) |
| `CHUNK_SIZE` | `300` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `3` | Maximum authorized candidates considered before relevance filtering |
| `SEARCH_K` | `10` | Nearest chunks initially retrieved from FAISS |
| `RELEVANCE_THRESHOLD` | `1.5` | Maximum accepted FAISS L2 distance (strictly less than this value) |
| `MAX_CONTEXT_CHUNKS` | `2` | Maximum authorized chunks included in the LLM prompt |
| `USE_LLM` | `False` | Whether to request an OpenAI-generated answer |

FAISS `IndexFlatL2` returns squared L2 distances. The threshold is therefore tied to this embedding and distance setup and may need calibration if the model or index changes.

## Access control and logging notes

The role check is applied to retrieved chunk metadata before chunks are assembled as LLM context. However, this is a demonstration implementation, not a secure authentication system:

- Usernames and roles are hard-coded in `USERS`; there are no passwords or identity provider.
- Raw retrieval matches, including document names, role labels, and chunk text, are printed **before** the role filter. Anyone who can see the terminal may therefore see restricted text. Remove or protect this diagnostic output before using sensitive documents.
- The access rules are hard-coded alongside document loading, and the current loader explicitly names the two sample files.
- `logs.jsonl` records the query, raw result text, authorized result text, user, role, and timestamp. Treat this file as sensitive; it is ignored by Git but is not encrypted or access-controlled by the program.
- The application does not provide tenant isolation, authorization storage, encrypted document storage, prompt-injection defenses, or a persistent vector database.

Do not use this prototype with confidential or regulated data without designing and reviewing the required security controls.

## Current limitations

- Command-line only; there is no HTTP API, web interface, or upload command.
- Documents and vectors are loaded and indexed at startup; updates require editing the files and restarting.
- Chunking is character-based and does not preserve document structure.
- Retrieval returns context rather than a generated answer unless LLM mode is enabled.
- Query logging occurs only when accessible results are found, and the LLM response is not currently logged.
- The distance threshold and retrieval settings are fixed in source code.

See [`blueprint.md`](blueprint.md) for the proposed next phases, including an API, stronger authentication, metadata storage, and retrieval enhancements.

## Dependencies

The project currently lists these packages in `requirements.txt`:

- `sentence-transformers` — local text embeddings
- `faiss-cpu` — vector similarity search
- `numpy` — embedding array conversion
- `requests` — listed dependency (not directly used by `main.py`)
- `openai` — optional answer generation

## License

No license is currently specified.
