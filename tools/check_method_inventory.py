#!/usr/bin/env python3
"""Validate the repo-local BaoIAD method inventory and documentation layout."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FIELDS = {'slug', 'display', 'family', 'config_paths', 'readme_path', 'alignment_path'}
FORBIDDEN_FIELD_NAMES = {'track', 'status', 'closed', 'caveat'}
PUBLIC_FORBIDDEN = (
    '../baoiad-paper', 'docs/evidence', 'paper-facing',
    'Track', 'Status', 'closed', 'caveat',
)
REQUIRED_README_SECTIONS = (
    '### MVTec AD result summary',
    '### VisA result summary',
    '### Speed summary',
    '### Alignment note',
)

@dataclass(frozen=True)
class Hit:
    path: Path
    label: str
    line: int
    text: str


def load_inventory(root: Path = ROOT):
    path = root / 'baoiad' / 'method_inventory.py'
    spec = importlib.util.spec_from_file_location('baoiad_method_inventory', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_inventory(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    module = load_inventory(root)
    methods = tuple(module.METHODS)
    if len(methods) != 37:
        errors.append(f'expected 37 methods, found {len(methods)}')
    families = tuple(module.families())
    if len(families) != 9:
        errors.append(f'expected 9 families, found {len(families)}')
    readmes = sorted((root / 'configs').glob('*/README.md'))
    if len(readmes) != 37:
        errors.append(f'expected 37 config READMEs, found {len(readmes)}')
    slugs = [m.slug for m in methods]
    if len(set(slugs)) != len(slugs):
        errors.append('duplicate method slugs')
    for entry in methods:
        fields = set(getattr(entry, '__dataclass_fields__', {}))
        if fields != ALLOWED_FIELDS:
            errors.append(f'{entry.slug}: inventory fields differ from allowed contract: {sorted(fields)}')
        if FORBIDDEN_FIELD_NAMES & fields:
            errors.append(f'{entry.slug}: forbidden inventory field present')
        if not (root / entry.readme_path).is_file():
            errors.append(f'{entry.slug}: missing README {entry.readme_path}')
        if not (root / entry.alignment_path).is_file():
            errors.append(f'{entry.slug}: missing alignment record {entry.alignment_path}')
        for cfg in entry.config_paths:
            if not (root / cfg).is_file():
                errors.append(f'{entry.slug}: missing config {cfg}')
        text = (root / entry.readme_path).read_text(encoding='utf-8')
        for section in REQUIRED_README_SECTIONS:
            if section not in text:
                errors.append(f'{entry.slug}: missing README section {section}')
        if entry.slug == 'pyramidflow' and 'Unavailable' not in _section(text, '### Speed summary'):
            errors.append('pyramidflow: speed summary must say Unavailable')
        align = (root / entry.alignment_path).read_text(encoding='utf-8') if (root / entry.alignment_path).is_file() else ''
        if entry.readme_path not in align:
            errors.append(f'{entry.slug}: alignment record does not link back to method README')
    return errors


def _section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ''
    nxt = text.find('\n### ', start + len(heading))
    return text[start:] if nxt < 0 else text[start:nxt]


def residual_hits(root: Path = ROOT, paths: tuple[str, ...] | None = None) -> list[Hit]:
    if paths is None:
        paths = ('README.md', 'README_zh-CN.md', 'docs', 'configs', 'tools/benchmark.py', 'tools/smoke_test_gpu.sh', 'tools/smoke_test_remaining.sh')
    files: list[Path] = []
    for raw in paths:
        p = root / raw
        if not p.exists():
            continue
        if p.is_dir():
            files.extend(x for x in p.rglob('*') if x.is_file())
        else:
            files.append(p)
    hits: list[Hit] = []
    skip_parts = {'.git', '.omx', '__pycache__', '.pytest_cache', '_build'}
    for file in files:
        rel = file.relative_to(root)
        if skip_parts & set(rel.parts):
            continue
        is_alignment_doc = rel.parts[:2] == ('docs', 'alignment')
        try:
            text = file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            terms = ('../baoiad-paper', 'docs/evidence', 'paper-facing') if is_alignment_doc else PUBLIC_FORBIDDEN
            for term in terms:
                if term in line:
                    hits.append(Hit(rel, term, lineno, line.strip()))
            if not is_alignment_doc and re.search(r'\|\s*strict\s*\|', line):
                hits.append(Hit(rel, 'strict-status-column', lineno, line.strip()))
    return hits


def main(argv: list[str] | None = None) -> int:
    errors = validate_inventory(ROOT)
    hits = residual_hits(ROOT)
    if hits:
        errors.extend(f'residual {h.label}: {h.path}:{h.line}: {h.text}' for h in hits)
    if errors:
        print('FAIL method inventory validation')
        for err in errors:
            print(f'- {err}')
        return 1
    print('PASS method inventory validation')
    print('method entries: 37')
    print('families: 9')
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
