import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = ("docs/en/vertical.md", "docs/es/vertical.md")
RUNNER = "Symbex1_PIO/lib/lib_symbex/tests/run_tests.py"
# This header is documented relative to the lib_symbex root, not the checkout.
RELATIVE_PATH_EXCEPTIONS = {
    "include/layers/conv/symbex_bool_vertical.h":
        "Symbex1_PIO/lib/lib_symbex/include/layers/conv/symbex_bool_vertical.h",
}


def _code_paths(text):
    for match in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
        path = match.group(1)
        if "/" in path and path.endswith((".py", ".c", ".h", ".md")):
            yield path


def _destination(root, path):
    return root / RELATIVE_PATH_EXCEPTIONS.get(path, path)


def test_vertical_docs_code_paths_exist():
    missing = []
    for doc in DOCS:
        paths = list(_code_paths((REPO / doc).read_text(encoding="utf-8")))
        if RUNNER not in paths:
            missing.append(f"{doc}: missing inline runner reference {RUNNER}")
        for path in paths:
            destination = _destination(REPO, path)
            if not destination.is_file():
                missing.append(f"{doc}: {path} -> {destination}")
    assert not missing, "Missing documented paths:\n" + "\n".join(missing)


def test_code_paths_filters_inline_spans():
    text = """`a/x.py` and `b/x.c` `c/x.h` `d/x.md`
`file.py` [link](a/link.md) `a/x.json` `identifier` `a/cmd.py --arg`
```python
```
"""
    assert list(_code_paths(text)) == ["a/x.py", "b/x.c", "c/x.h", "d/x.md"]


def test_header_exception_still_requires_a_file(tmp_path):
    header = "include/layers/conv/symbex_bool_vertical.h"
    destination = _destination(tmp_path, header)
    assert destination == tmp_path / RELATIVE_PATH_EXCEPTIONS[header]
    assert not destination.is_file()
    destination.parent.mkdir(parents=True)
    destination.write_text("", encoding="utf-8")
    assert destination.is_file()
