"""PreToolUse guard: blocks shell commands that would bypass the maintainer's rules.

Exit code 2 blocks the tool call and shows the reason to Claude.
"""

import json
import re
import shutil
import subprocess
import sys

SPLIT = re.compile(r"&&|\|\||[;|\n]")

RULES = [
    (r"\bgh\s+pr\s+merge\b", "Only the maintainer merges PRs."),
    (r"--no-verify\b", "Skipping hooks is not allowed."),
    (
        r"\bgit\b.*\bpush\b.*(--force\b|--force-with-lease|\s-f\b|\s\+\S)",
        "Force push is not allowed.",
    ),
    (r"\bgit\b.*\bpush\b.*(\bmain\b|\bmaster\b)", "Pushing to main is not allowed."),
    (
        r"\bgit\b.*\bconfig\b.*\buser\.(email|name)\s+[^\s|;&]",
        "Changing git identity is not allowed.",
    ),
    (
        r"\bgh\s+repo\s+(delete|edit\b.*--visibility)",
        "Deleting the repo or changing visibility needs the maintainer.",
    ),
    (
        r"\bgh\s+api\b.*(-X|--method)\s*DELETE\b",
        "DELETE calls to the GitHub API need the maintainer.",
    ),
    (
        r"\bgh\s+api\b.*rulesets.*(-X|--method)\s*(PUT|PATCH)\b",
        "Editing branch rulesets needs the maintainer.",
    ),
]


def current_branch(cwd: str | None) -> str:
    git = shutil.which("git")
    if git is None:
        return ""
    try:
        # Fixed args, no shell, so S603 is safe here.
        out = subprocess.run(  # noqa: S603
            [git, "symbolic-ref", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def check(command: str, cwd: str | None) -> str | None:
    for seg in SPLIT.split(command):
        seg = seg.strip()
        if not seg:
            continue
        for pattern, reason in RULES:
            if re.search(pattern, seg):
                return reason
        on_main = None
        if re.search(r"\bgit\b.*\b(commit|push)\b", seg):
            on_main = current_branch(cwd) in ("main", "master")
        if on_main and re.search(r"\bgit\b.*\bcommit\b", seg):
            return "Committing on main is not allowed. Create a feature branch."
        push = re.search(r"\bgit\b.*\bpush\b(.*)", seg)
        if on_main and push:
            args = [a for a in push.group(1).split() if not a.startswith("-")]
            if len(args) < 2 or any(a == "HEAD" or a.startswith("HEAD:") for a in args):
                return "Pushing from main is not allowed. Name the feature branch."
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = data.get("tool_input") or {}
    command = tool_input.get("command") or tool_input.get("cmd") or ""
    if not isinstance(command, str) or not command:
        return 0
    reason = check(command, data.get("cwd"))
    if reason:
        print(f"Blocked by project guard: {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
