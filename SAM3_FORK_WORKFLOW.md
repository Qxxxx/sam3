# SAM3 Fork Workflow

This document defines how to work with the `sam3` submodule fork safely.

## Branch Roles

- `main`: mirror of `upstream/main` only. Do not add custom commits here.
- `custom/main`: long-lived integration branch for fork-specific changes.
- `feat/*`: short-lived feature branches created from `custom/main`.

## Remotes

In `sam3`:

- `origin` = your fork (`git@github.com:Qxxxx/sam3.git`)
- `upstream` = original repo (`git@github.com:facebookresearch/sam3.git`)
- push to `upstream` is disabled by config.

## Daily Development

```bash
cd /Users/qiguang/workspace/duolian_pose/sam3
git checkout custom/main
git pull --rebase

git checkout -b feat/<topic>
# make changes

git add -A
git commit -m "feat(<scope>): <summary>"
git push -u origin feat/<topic>
```

## Sync With Upstream

```bash
cd /Users/qiguang/workspace/duolian_pose/sam3

git checkout main
git fetch upstream --prune
git merge --ff-only upstream/main
git push origin main

git checkout custom/main
git rebase main
git push --force-with-lease origin custom/main
```

## PR Rules

- Fork internal integration PRs: base = `custom/main`.
- PR to original repo: branch from `main`, base = `upstream/main`, keep diff minimal.
- PR body should include:
  - Summary
  - Testing commands and results
  - Screenshots (or "Not applicable")
  - Related issues

## Commit Rules

- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `chore:`.
- One commit = one logical change.
- Do not mix refactor/format-only changes with feature logic in one commit.
- Keep commit messages specific (avoid generic messages like "update" or "fix stuff").

## Rebase Pain Mitigation

- Keep feature branches short-lived.
- Rebase frequently (small conflict sets are easier).
- Squash noisy WIP commits before opening PR.
- `rerere` is enabled to reuse conflict resolutions.

## Submodule Reminder

After new commits are added in `sam3`, commit the submodule pointer in the parent repo:

```bash
cd /Users/qiguang/workspace/duolian_pose
git add sam3 .gitmodules
git commit -m "chore: update sam3 submodule"
```

Without this parent commit, other collaborators will not receive the updated submodule commit.
