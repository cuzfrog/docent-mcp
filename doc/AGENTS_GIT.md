## Git 
- Main branch: `main`
- By default, use `scripts/gh-bot.mjs` as a `gh` wrapper to give you an identity; unless the user asks you to merge PR via `gh` on their behalf.

### PR title - semantic-pull-request format:
<type>([optional task_id]): <description>
```yaml
types:
  - feat
  - fix
  - docs
  - style
  - refactor
  - perf
  - test
  - chore
  - ci
```

### PR Squash commit message
- Do not contain any individual commits. Use the github template.
