"""Deterministic numerical and portability checks for the public release."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bmts


def main() -> None:
    checks = bmts.run_self_checks()
    for name, value in checks.items():
        assert value < 1e-9, f"{name} failed: {value}"

    source = ROOT / "data" / "nestorowa_bmts_input.npz"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == "bba96dd4a725990fab0a54cde49571bf97316466d85e6595dd47864442fee302"
    data = bmts.load_nestorowa(source)
    assert data["rna_pcs"].shape == (1656, 9)
    assert data["facs"].shape[0] == 1656
    observed = np.all(np.isfinite(data["facs"]), axis=1)
    assert int(observed.sum()) == 1430
    assert int((~observed).sum()) == 226

    metadata = json.loads((ROOT / "data" / "nestorowa_source.json").read_text())
    assert metadata["artifact"]["sha256"] == digest
    assert metadata["dataset"]["license"] == "CC0"

    text_suffixes = {
        "",
        ".cff",
        ".gitignore",
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
    }
    absolute_path_patterns = (
        re.compile(re.escape("/" + "Users" + "/") + r"[^/\s]+/"),
        re.compile(re.escape("/" + "home" + "/") + r"[^/\s]+/"),
        re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    )
    workspace_fragments = (
        "Dropbox-" + "BaruchCollege",
        "Projects-" + "1Works",
        ".codex/" + "visualizations",
    )
    credential_patterns = (
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
            r"\s*[:=]\s*[\"'][^\"']{8,}[\"']"
        ),
    )
    disallowed_directories = {
        ".ipynb_checkpoints",
        ".matplotlib",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "local",
        "logs",
        "venv",
    }
    disallowed_names = {
        ".env",
        ".env.local",
        "credentials.json",
        "id_rsa",
        "id_ed25519",
        "secrets.json",
    }
    disallowed_raw_suffixes = {
        ".db",
        ".h5",
        ".hdf5",
        ".key",
        ".p12",
        ".parquet",
        ".pem",
        ".pfx",
        ".rdata",
        ".rds",
        ".sqlite",
    }

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        assert not any(part in disallowed_directories for part in relative.parts), path
        if not path.is_file():
            continue
        assert path.name.lower() not in disallowed_names, path
        assert path.suffix.lower() not in disallowed_raw_suffixes, path
        assert path.stat().st_size <= 5 * 1024 * 1024, path
        if path.suffix.lower() in text_suffixes or path.name == ".gitignore":
            text = path.read_text(encoding="utf-8")
            assert not any(pattern.search(text) for pattern in absolute_path_patterns), path
            assert not any(fragment in text for fragment in workspace_fragments), path
            assert not any(pattern.search(text) for pattern in credential_patterns), path

    expected_data_files = {
        "README.md",
        "nestorowa_bmts_input.npz",
        "nestorowa_source.json",
        "paper_scale_summary.json",
    }
    assert {path.name for path in (ROOT / "data").iterdir()} == expected_data_files

    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 4
    for path in notebook_paths:
        notebook = json.loads(path.read_text())
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        assert [cell["execution_count"] for cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        )
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )

    print("BMTS numerical, data-integrity, and public-tree checks passed.")
    for name, value in checks.items():
        print(f"  {name}: {value:.3e}")


if __name__ == "__main__":
    main()
