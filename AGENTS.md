# AGENTS.md

You are working in the `ece405-assignment4-alignment` repository. Treat this file as operating instructions for the agent.

## Start Of Session

- Read this file before making changes.
- Prefer local evidence from the repo over assumptions.
- Inspect the specific files, scripts, tests, and logs relevant to the current task before editing.
- If you change the workflow, orchestration, logging layout, or output conventions, update this file before you finish.

## Python And Commands

- Use `uv run ...` for Python commands unless there is a concrete reason not to.
- Use `uv sync` after dependency changes.
- Use `uv run pytest` for the full test suite.
- Use `test_and_make_submission.sh` only when you specifically need the submission helper.

## Validation Pipeline

- Start from the smallest discriminating check, not the largest rerun.
- Before broad experiments, prefer one of these cheap validations when applicable:
	- `uv run python -m py_compile <touched_file>`
	- a narrow `uv run pytest ...` target for the touched logic
	- a smoke-test mode on the relevant script
	- replaying the smallest failed artifact slice if debugging runtime or memory issues
- After the first substantive code edit, run one focused validation immediately before doing more broad work.
- If a narrow validation fails, repair that slice and rerun the same validation before expanding scope.
- Use full-suite runs only after the touched slice is locally stable, or when the user explicitly asks for it.

## Background Run Pipeline

- For long-running training, evaluation, or sweep jobs, prefer a detached `tmux` session rather than tying the run to an interactive shell.
- Before starting a new background run, check whether a session for the same task or campaign is already active.
- Prefer a small smoke test before launching the full detached run.
- When a task has multiple configs, prefer a dedicated orchestrator or sweep script over manual repeated commands.
- If a wrapper script prints identifiers like a `session=` name or a `campaign=` name, capture and reuse them when monitoring logs and artifacts.

## Logging And Output Expectations

- Keep human-readable logs under `logs/<area>/<campaign>/` when the task has a natural area or campaign grouping.
- Keep sweep-level or orchestration progress in an `orchestrator.log` at the campaign root when a run launches multiple configs.
- Keep per-run logs alongside the orchestrator log with descriptive names.
- Keep machine-readable outputs under `data/<area>/<experiment_or_campaign>/<run_name>/`.
 - Keep derived reports and plots under `data/<area>/analysis/<campaign>/` or the matching analysis directory for that task.

## Monitoring And Debugging

- When monitoring a background run, read `orchestrator.log` first if it exists, then inspect the active per-run log.
- Use on-disk artifacts as the source of truth for where a run stopped.
- If a run dies mid-pipeline, identify the last completed artifact and the first missing expected artifact before hypothesizing about root cause.
- When debugging memory or stability issues, reproduce the smallest failing slice instead of relaunching the whole workflow.
- Suspect cleanup and phase-boundary memory retention when a run consistently survives early iterations and dies later without a Python traceback.

## Reporting And Closeout

- If the repo already has a report-generation script for the task, run or verify it after the experiment finishes.
- Confirm that plots, summaries, and machine-readable outputs were written to the expected analysis directory.
- In the final user-facing summary, report what changed, what was validated, what is still running, and any concrete next monitoring step.

## Minimal Startup Prompt For Future Agents

Use the following prompt when bootstrapping a new agent for work in this repo:

```text
Read AGENTS.md first and follow it strictly.

You are working in the ece405-assignment4-alignment repository.

Required startup steps:
1. Read AGENTS.md and inspect the files and scripts relevant to the current task.
2. Check whether a related tmux session or long-running background job is already active before launching anything new.
3. Inspect the relevant log directory and read orchestrator.log first when it exists.
4. Use machine-readable artifacts under data/... and logs under logs/... as the source of truth.
5. Before broad reruns, prefer the smallest discriminating validation:
   - uv run python -m py_compile <touched_file>
   - a narrow uv run pytest target
   - a smoke-test mode on the touched script
   - replay of the exact failed artifact slice if debugging runtime or memory issues
6. After long runs finish, run or verify the matching report-generation step if one exists and confirm outputs on disk.

Output rules:
- Keep logs under logs/...
- Keep run artifacts under data/...
- Keep reports under the matching analysis directory
- If you change workflow or orchestration, update AGENTS.md before finishing.
```