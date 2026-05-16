import csv
import json
import shutil
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------------------------
def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _to_path(path):
    return path if isinstance(path, Path) else Path(path)


def _safe_name(text):
    # filesystem-safe name, spaces and slashes become underscores
    keep = []
    for ch in str(text).strip():
        if ch.isalnum() or ch in ('_', '-', '.'):
            keep.append(ch)
        elif ch in (' ', '/'):
            keep.append('_')
    name = ''.join(keep).strip('_')
    return name or 'unnamed'


def _portable_path_str(path_like, fair_root):
    # absolute path inside fair_root -> relative form, everything else -> basename
    # no .resolve() calls so the result does not depend on cwd
    p = _to_path(path_like)
    root = _to_path(fair_root)
    if p.is_absolute() and root.is_absolute():
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return p.name
    return p.name


def _git_commit(cwd=None):
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=str(cwd) if cwd else None,
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except Exception:
        return None


def sha256sum(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with _to_path(path).open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_fair_tree(root_dir):
    # create the standard folder layout, return the path dict
    root = _to_path(root_dir)
    tree = {
        'root': root,
        'raw': root / 'raw',
        'processed': root / 'processed',
        'derived': root / 'derived',
        'metadata': root / 'metadata',
        'manifests': root / 'manifests',
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def _dataset_filename(plot_name, dataset_label, level, extension='.csv'):
    return f'DataSet_{_safe_name(plot_name)}_{_safe_name(dataset_label)}_{_safe_name(level)}{extension}'


# ------------------------------------------------------------------------------------------------------------------
# Main export functions
# ------------------------------------------------------------------------------------------------------------------
def save_dataset(data, fair_root_dir, plot_name, dataset_label, level='derived', description=None,
                 units=None, source_files=None, source_script=None, processing_summary=None,
                 fit_model=None, fit_parameters=None, extra_metadata=None, index=False):
    # write one dataset as CSV + JSON sidecar; level is raw, processed or derived
    # returns (csv_path, metadata_json_path)
    tree = ensure_fair_tree(fair_root_dir)
    level_key = _safe_name(level).lower()
    if level_key not in ('raw', 'processed', 'derived'):
        raise ValueError("level must be one of: raw, processed, derived")

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    filename = _dataset_filename(plot_name, dataset_label, level_key, '.csv')
    csv_path = tree[level_key] / filename
    df.to_csv(csv_path, index=index)

    columns_meta = []
    for col in df.columns:
        col_s = str(col)
        columns_meta.append({'name': col_s, 'dtype': str(df[col].dtype),
                             'unit': (units or {}).get(col_s)})

    source_files_portable = [_portable_path_str(s, tree['root']) for s in (source_files or [])]

    metadata = {
        'dataset_name': filename.removesuffix('.csv'),
        'plot_name': plot_name,
        'dataset_label': dataset_label,
        'level': level_key,
        'created_utc': _utc_now_iso(),
        'n_rows': int(len(df)),
        'n_columns': int(len(df.columns)),
        'columns': columns_meta,
        'description': description,
        'source_files': source_files_portable,
        'source_script': _portable_path_str(source_script, tree['root']) if source_script else None,
        'processing_summary': processing_summary,
        'fit_model': fit_model,
        'fit_parameters': fit_parameters or {},
        'sha256': sha256sum(csv_path),
        'git_commit': _git_commit(cwd=Path.cwd()),
    }
    if extra_metadata:
        metadata['extra_metadata'] = extra_metadata

    meta_path = tree['metadata'] / filename.replace('.csv', '.json')
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    _upsert_dataset_manifest_row(tree, metadata, csv_path, meta_path)
    return csv_path, meta_path


def append_provenance(fair_root_dir, dataset_name, was_derived_from, processing_step,
                      script_path=None, function_name=None, parameters=None):
    # one row appended to metadata/provenance.csv per call
    tree = ensure_fair_tree(fair_root_dir)
    out = tree['metadata'] / 'provenance.csv'
    portable_sources = [_portable_path_str(s, tree['root']) for s in was_derived_from]
    row = {
        'created_utc': _utc_now_iso(),
        'dataset_name': dataset_name,
        'was_derived_from': json.dumps(portable_sources, ensure_ascii=False),
        'processing_step': processing_step,
        'script_path': _portable_path_str(script_path, tree['root']) if script_path else '',
        'function_name': function_name or '',
        'parameters': json.dumps(parameters or {}, ensure_ascii=False),
        'git_commit': _git_commit(cwd=Path.cwd()) or '',
    }
    _append_manifest_row(out, row)
    return out


def write_checksums_manifest(fair_root_dir):
    # hash every file in the package, skip only the checksums file itself
    tree = ensure_fair_tree(fair_root_dir)
    out = tree['manifests'] / 'checksums_sha256.csv'
    rows = []
    for path in sorted(tree['root'].rglob('*')):
        if not path.is_file() or path == out:
            continue
        rel = path.relative_to(tree['root']).as_posix()
        # first path component is raw/processed/derived/metadata/manifests, top-level files become 'root'
        level = rel.split('/', 1)[0] if '/' in rel else 'root'
        rows.append({'level': level, 'path': rel, 'sha256': sha256sum(path)})
    _write_rows(out, rows)
    return out


def write_readme(fair_root_dir, project_name, description=None, contact=None,
                 extra_sections=None, include_manifest_summary=True):
    # write README.md into the FAIR root folder. extra_sections is an optional
    # list of (heading, markdown_body) pairs the caller wants appended verbatim.
    tree = ensure_fair_tree(fair_root_dir)
    readme_path = tree['root'] / 'README.md'

    dataset_count = 'unknown'
    if include_manifest_summary:
        ds_manifest = tree['manifests'] / 'datasets_manifest.csv'
        if ds_manifest.exists():
            try:
                dataset_count = str(len(pd.read_csv(ds_manifest)))
            except Exception:
                pass

    lines = [f'# {project_name}', '']
    if description:
        lines.extend([description, ''])
    lines.append(f'_Generated (UTC): {_utc_now_iso()}_')
    if contact:
        lines.append(f'_Contact: {contact}_')
    lines.append('')

    lines.extend([
        '## Contents',
        '',
        '- `raw/` original measurement CSVs, deduplicated by SHA-256 checksum.',
        '  Layout: `raw/<campaign>/<topic>/<date>/<sha12>__<filename>`.',
        '- `processed/` data after the first loading step (dark subtraction, signal conversion), before any plot-specific reduction.',
        '- `derived/` the numbers actually plotted in the figure, plus fit summaries.',
        '- `metadata/` one JSON sidecar per dataset (schema, units, sources, sha256) and `provenance.csv`.',
        '- `manifests/` dataset index, raw-file index, and `checksums_sha256.csv` for the whole package.',
        '',
        '## Notes',
        '',
        'Paths inside the sidecars and manifests are relative to this folder. Anything that lived outside (the analysis script, '
        'external mapping tables) is stored by basename only.',
        '',
        'Each sidecar records the git commit of the analysis code at export time. Checking out that commit in the source '
        'repository reproduces the derived datasets from the staged raw inputs.',
        '',
        f'{dataset_count} datasets in the manifest.',
    ])

    for heading, body in (extra_sections or []):
        lines.extend(['', f'## {heading}', '', body])

    readme_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return readme_path


def stage_raw_files(fair_root_dir, source_files, campaign, topic, date_label=None):
    # copy raw inputs into raw/<campaign>/<topic>/<date>/<sha12>__<filename>
    # one copy per checksum, every reference logged in manifests/raw_file_index.csv
    tree = ensure_fair_tree(fair_root_dir)

    # drop duplicates already present on the caller list
    source_unique = []
    seen_sources = set()
    for src in source_files:
        p = _to_path(src).resolve()
        if p not in seen_sources:
            source_unique.append(p)
            seen_sources.add(p)

    target_root = (tree['raw'] / _safe_name(campaign) / _safe_name(topic) /
                   (_safe_name(date_label) if date_label else 'undated'))
    target_root.mkdir(parents=True, exist_ok=True)

    # reuse files already staged in a previous run, looked up by checksum
    raw_index_path = tree['manifests'] / 'raw_file_index.csv'
    existing_by_checksum = {}
    if raw_index_path.exists():
        try:
            df_idx = pd.read_csv(raw_index_path)
            if 'sha256' in df_idx.columns and 'staged_path' in df_idx.columns:
                for _, row in df_idx.iterrows():
                    checksum = str(row['sha256'])
                    staged_path = str(row['staged_path'])
                    if checksum and staged_path:
                        existing_by_checksum[checksum] = staged_path
        except Exception:
            pass

    staged_paths = []
    for src in source_unique:
        if not src.exists() or not src.is_file():
            continue
        checksum = sha256sum(src)
        staged = None
        if checksum in existing_by_checksum:
            candidate = Path(existing_by_checksum[checksum])
            if not candidate.is_absolute():
                candidate = tree['root'] / candidate
            if candidate.exists():
                staged = candidate
        if staged is None:
            staged_name = f'{checksum[:12]}__{_safe_name(src.name)}'
            staged = target_root / staged_name
            if not staged.exists():
                shutil.copy2(src, staged)
            existing_by_checksum[checksum] = _portable_path_str(staged, tree['root'])

        staged_paths.append(staged)
        _append_manifest_row(raw_index_path, {
            'created_utc': _utc_now_iso(),
            'campaign': campaign,
            'topic': topic,
            'date_label': date_label or '',
            'source_path': str(src),
            'source_name': src.name,
            'source_size_bytes': src.stat().st_size,
            'sha256': checksum,
            'staged_path': _portable_path_str(staged, tree['root']),
            'git_commit': _git_commit(cwd=Path.cwd()) or '',
        })

    return staged_paths


# ------------------------------------------------------------------------------------------------------------------
# Internal CSV helpers
# ------------------------------------------------------------------------------------------------------------------
def _upsert_dataset_manifest_row(tree, metadata, csv_path, meta_path):
    # upsert by dataset_name so the manifest stays clean across re-runs
    manifest_path = tree['manifests'] / 'datasets_manifest.csv'
    row = {
        'created_utc': metadata['created_utc'],
        'dataset_name': metadata['dataset_name'],
        'level': metadata['level'],
        'csv_path': _portable_path_str(csv_path, tree['root']),
        'metadata_path': _portable_path_str(meta_path, tree['root']),
        'sha256': metadata['sha256'],
        'n_rows': metadata['n_rows'],
        'n_columns': metadata['n_columns'],
    }
    fieldnames = list(row.keys())

    rows = []
    if manifest_path.exists():
        try:
            with manifest_path.open('r', encoding='utf-8') as f:
                for r in csv.DictReader(f):
                    if r.get('dataset_name') == row['dataset_name']:
                        continue  # the new row replaces this one
                    rows.append({k: r.get(k, '') for k in fieldnames})
        except Exception:
            rows = []
    rows.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _append_manifest_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
