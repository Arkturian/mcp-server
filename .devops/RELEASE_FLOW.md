# 🚀 Release Flow Overview

## Branches
- **`dev`** – Integration branch for day-to-day work.
- **`main`** – Production branch; pushing here triggers the deployment workflow.

## Typical Release Steps
1. Develop features on short-lived branches branched off `dev`.
2. Merge into `dev` via pull request. CI (`dev.yml`) runs automatically.
3. When ready to ship, ensure `dev` is green and run `.devops/scripts/release.sh`.
4. The script fast-forwards `main`, pushes to GitHub, and GitHub Actions (`deploy.yml`) handles build + deploy.
5. Monitor the workflow run with `gh run watch` or via the Actions tab.
6. Validate the site at `https://mcp.arkturian.com` (if applicable) and confirm fresh assets reached `/var/www/mcp.arkturian.com`.

## Release to one target

For a reviewed integration branch that must reach only one host, push its clean
HEAD and run:

```bash
./devops release --target arkturian
```

This dispatches `deploy.yml` for the current branch, with the expected full SHA,
without promoting `main`. Only the selected host runs; `pdrei` and `all` are
also explicit target options. The workflow rejects a moved ref before SSH and
deploys its exact SHA rather than fetching the latest `main`. An ordinary
release without `--target` retains the existing main-promotion and both-host
deployment behavior.

Check the workflow's head SHA, target job, skipped other job, production Git
HEAD and `/health` after release. A successful dispatch alone is not a successful
deployment. Do not use `.devops/servers/arkserver.yaml` for MCP: it is an unrelated
Storage template.

## Rollback Strategy
- Use `.devops/rollback.sh` on the server to restore from `/var/backups/mcp-server-*`.
- Alternatively revert the offending commit on `main` and push to trigger a redeploy.
