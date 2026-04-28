# No-Semantics Cleanup

Use this workflow for cleanup that must not change behavior.

## Scope

- Remove placeholder comments.
- Remove unused imports.
- Remove stale helpers only after confirming no references remain.
- Tighten small documentation mismatches.
- Keep diffs small and mechanical.

## Do Not Change

- observation shape
- action space or masks
- PPO
- GAE
- reward semantics
- weapon semantics
- recurrent hidden state handling
- actor or critic structure
- opponent behavior
- key diagnostics

## Checklist

- Locate all references before deleting or renaming.
- Avoid formatting churn outside touched lines.
- Keep cleanup separate from semantic fixes.
- Run the narrowest relevant pytest target.
- If docs mention removed helpers or changed entrypoints, update docs in the same cleanup.

## Output

- List removed or simplified code.
- State why behavior is unchanged.
- Include validation commands.
