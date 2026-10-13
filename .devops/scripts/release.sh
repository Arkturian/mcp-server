#!/usr/bin/env bash

set -euo pipefail

# Resolve repository root relative to this script so it works anywhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
DEV_BRANCH="dev"
MAIN_BRANCH="main"
CHECKOUT_SCRIPT="$SCRIPT_DIR/checkout-branch.sh"
BUILD_SCRIPT="$SCRIPT_DIR/build-local.sh"

usage() {
  cat <<'USAGE'
Usage: release.sh [--no-build] [--target arkturian|pdrei|all]

Fast-forwards main from dev, pushes to origin, and
triggers the GitHub Actions deployment workflow. By default the local build
script runs beforehand.

Options:
  --no-build   Skip running the local build before pushing
  --target     Dispatch only the named target at this branch's pushed HEAD.
               Does not promote main. Requires a clean, synchronized branch.
USAGE
}

run_build=true
release_target=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      run_build=false
      shift
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "Error: --target requires arkturian, pdrei or all." >&2
        exit 1
      fi
      release_target="$2"
      case "$release_target" in arkturian|pdrei|all) ;; *) usage; exit 1;; esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree has uncommitted changes. Please commit or stash them first." >&2
  exit 1
fi

if [[ -n "$release_target" ]]; then
  release_branch="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "Error: targeted release requires a named branch." >&2
    exit 1
  }
  release_revision="$(git rev-parse HEAD)"
  remote_revision="$(git ls-remote --exit-code origin "refs/heads/$release_branch" | awk '{print $1}')"
  if [[ "$remote_revision" != "$release_revision" ]]; then
    echo "Error: push the reviewed branch before a targeted release." >&2
    exit 1
  fi
  if [[ "$run_build" == true ]]; then
    "$BUILD_SCRIPT"
  fi
  # The workflow compares this SHA with its resolved ref before any SSH action.
  # A concurrent branch update fails closed, rather than deploying other code.
  gh workflow run deploy.yml --ref "$release_branch" \
    --field "target=$release_target" --field "revision=$release_revision"
  echo "Targeted release dispatched: $release_target at $release_revision."
  exit 0
fi

printf '\n==> Syncing %s branch\n' "$DEV_BRANCH"
"$CHECKOUT_SCRIPT" "$DEV_BRANCH"

git push origin "$DEV_BRANCH"

if [[ "$run_build" == true ]]; then
  printf '\n==> Running local production build\n'
  "$BUILD_SCRIPT"
fi

printf '\n==> Syncing %s branch\n' "$MAIN_BRANCH"
"$CHECKOUT_SCRIPT" "$MAIN_BRANCH"

printf '\n==> Fast-forwarding %s from %s\n' "$MAIN_BRANCH" "$DEV_BRANCH"
if ! git merge --ff-only "$DEV_BRANCH"; then
  echo "Error: $MAIN_BRANCH cannot fast-forward from $DEV_BRANCH. Resolve manually (likely $MAIN_BRANCH is ahead)." >&2
  exit 1
fi

git push origin "$MAIN_BRANCH"

printf '\n✅ Release pushed. GitHub Actions will build & deploy from %s.\n' "$MAIN_BRANCH"

printf '\n==> Switching back to %s\n' "$DEV_BRANCH"
"$CHECKOUT_SCRIPT" "$DEV_BRANCH"
