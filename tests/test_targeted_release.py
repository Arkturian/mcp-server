"""Exercise the real release script with local Git and a non-deploying gh stub."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / '.devops/scripts/release.sh'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def release_repo(tmp_path):
    repo = tmp_path / 'repo'
    remote = tmp_path / 'origin.git'
    repo.mkdir()
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    git(repo, 'init')
    git(repo, 'config', 'user.email', 'fixture@example.test')
    git(repo, 'config', 'user.name', 'Release fixture')
    git(repo, 'checkout', '-b', 'feature/reviewed')
    scripts = repo / '.devops/scripts'
    scripts.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts / 'release.sh')
    (scripts / 'build-local.sh').write_text('#!/bin/bash\nprintf built > "$BUILD_CAPTURE"\n')
    (scripts / 'build-local.sh').chmod(0o755)
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'fixture')
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'push', '-u', 'origin', 'HEAD')
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'gh').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$RELEASE_CAPTURE"\n')
    (bindir / 'gh').chmod(0o755)
    capture = tmp_path / 'dispatch.txt'
    build = tmp_path / 'build.txt'
    env = {**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH'],
           'RELEASE_CAPTURE': str(capture), 'BUILD_CAPTURE': str(build)}

    def run(*args):
        return subprocess.run(['bash', str(scripts / 'release.sh'), *args], cwd=repo,
                              env=env, capture_output=True, text=True)

    return repo, remote, capture, build, run


@pytest.mark.parametrize('skip_build', [False, True])
def test_dispatch_pins_pushed_head_without_promoting_main(release_repo, skip_build):
    repo, remote, capture, build, run = release_repo
    head = git(repo, 'rev-parse', 'HEAD')
    refs = git(remote, 'show-ref')
    args = ['--target', 'arkserver'] + (['--no-build'] if skip_build else [])
    result = run(*args)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        'workflow', 'run', 'deploy.yml', '--ref', 'feature/reviewed',
        '--field', 'target=arkserver', '--field', 'revision=' + head,
    ]
    assert build.exists() is not skip_build
    assert git(remote, 'show-ref') == refs
    assert git(repo, 'branch', '--show-current') == 'feature/reviewed'
    assert git(repo, 'status', '--porcelain') == ''


@pytest.mark.parametrize('state', ['dirty', 'unpushed', 'detached', 'remote_advanced'])
def test_refuses_unreviewable_state_without_dispatch(release_repo, state):
    repo, remote, capture, build, run = release_repo
    if state == 'dirty':
        (repo / 'untracked').write_text('unfinished')
    elif state == 'detached':
        git(repo, 'checkout', '--detach')
    else:
        old = git(repo, 'rev-parse', 'HEAD')
        git(repo, 'commit', '--allow-empty', '-m', 'later')
        if state == 'remote_advanced':
            git(repo, 'push', 'origin', 'HEAD')
            git(repo, 'reset', '--hard', old)
    result = run('--target', 'arkserver')
    assert result.returncode != 0
    assert not capture.exists()
    assert not build.exists()


@pytest.mark.parametrize('args', [('--target',), ('--target', 'unknown')])
def test_invalid_target_never_dispatches(release_repo, args):
    _, _, capture, _, run = release_repo
    assert run(*args).returncode != 0
    assert not capture.exists()
