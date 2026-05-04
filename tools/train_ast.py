#!/usr/bin/env python3
"""Two-stage AST training entry point."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Train AST with teacher -> student stages.')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Working directory to save logs and models')
    parser.add_argument('--resume', action='store_true', help='Resume each stage from latest checkpoint if present')
    parser.add_argument('--cpu', action='store_true', help='Force CPU device')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        default=None,
        help='Override config options in key=value format.',
    )
    return parser.parse_args()


def _find_last_checkpoint(work_dir: str) -> str:
    last_path = os.path.join(work_dir, 'last_checkpoint')
    if os.path.exists(last_path):
        with open(last_path) as handle:
            return handle.read().strip()

    epoch_paths = sorted(glob.glob(os.path.join(work_dir, 'epoch_*.pth')))
    if not epoch_paths:
        raise FileNotFoundError(f'No checkpoint found under {work_dir}')
    return epoch_paths[-1]


def _run_stage(stage_name: str, config: str, work_dir: str, args, extra_cfg_options: list[str]) -> None:
    python = sys.executable
    train_script = os.path.join(ROOT, 'tools', 'train.py')
    cmd = [python, train_script, config, '--work-dir', work_dir]
    if args.cpu:
        cmd.insert(2, '--cpu')
    if args.resume:
        cmd.append('--resume')
    cfg_options = list(args.cfg_options or [])
    cfg_options.extend(extra_cfg_options)
    if cfg_options:
        cmd.extend(['--cfg-options', *cfg_options])

    print(f'[AST] Stage={stage_name} work_dir={work_dir}', flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    args = parse_args()
    work_dir = args.work_dir or os.path.join(ROOT, 'work_dirs', 'ast')
    teacher_dir = os.path.join(work_dir, 'teacher')
    student_dir = os.path.join(work_dir, 'student')

    common_checkpoint_opts = [
        'default_hooks.checkpoint.save_last=True',
    ]

    _run_stage(
        'teacher',
        args.config,
        teacher_dir,
        args,
        common_checkpoint_opts + ['model.training_phase=teacher'],
    )
    teacher_checkpoint = _find_last_checkpoint(teacher_dir)

    _run_stage(
        'student',
        args.config,
        student_dir,
        args,
        common_checkpoint_opts + [
            'model.training_phase=student',
            f'model.teacher_checkpoint={teacher_checkpoint}',
        ],
    )

    print(f'[AST] Teacher checkpoint: {teacher_checkpoint}', flush=True)
    print(f'[AST] Student work dir: {student_dir}', flush=True)


if __name__ == '__main__':
    main()
