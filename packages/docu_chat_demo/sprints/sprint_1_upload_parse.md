# Sprint 1: Upload and Parse

**Goal:** Uploaded PDF, TXT and DOCX files are parsed to plain text, and the
character count per file is visible in the UI.

**Time:** 10 minutes

---

## Phase 1.1 - Text Extraction

- [ ] In `core.py`, add `extract_text(name: str, data: bytes) -> str`
- [ ] Dispatch on the lowercased file extension — `.pdf`, `.docx`, `.txt`
- [ ] PDF: `pypdf.PdfReader(io.BytesIO(data))`, join `page.extract_text()`
      across pages with `"\n"`
- [ ] DOCX: `docx.Document(io.BytesIO(data))`, join paragraph `.text`
- [ ] TXT: `data.decode("utf-8", errors="replace")`
- [ ] Unsupported extension → raise `ValueError` with the filename

> Everything stays in memory — `io.BytesIO`, never a temp file. That removes
> the whole cleanup and path-traversal surface in one move.

## Phase 1.2 - Wire the Uploader

- [ ] In `app.py`, loop over `st.file_uploader(...)` results
- [ ] For each: call `extract_text(f.name, f.getvalue())` inside `try/except`
- [ ] Empty text (scanned PDF) → `st.warning` naming the file, skip it
- [ ] Exception → `st.warning` naming the file and the reason, keep going
- [ ] Show `st.write(f"{f.name}: {len(text)} chars")` per successful file
- [ ] Keep the good results as a `list[tuple[str, str]]` of `(name, text)`

> Partial success is the rule: one bad file must not stop the others.

---

## Expected output

Uploading three files prints one line per file with a non-zero character
count. A renamed or corrupt file shows a warning and does not crash the app.

## Verify

```bash
uv run streamlit run packages/docu_chat_demo/src/docu_chat_demo/app.py
```

1. Upload one `.pdf`, one `.txt`, one `.docx` → three lines, all counts > 0.
2. Rename a `.txt` to `.pdf` and upload it → warning, no traceback, app alive.

Quick non-UI check of the parser alone:

```bash
uv run python -c "
from pathlib import Path
import sys; sys.path.insert(0,'packages/docu_chat_demo/src')
from docu_chat_demo.core import extract_text
p = Path('YOUR_TEST.pdf')
print(len(extract_text(p.name, p.read_bytes())), 'chars')
"
```
