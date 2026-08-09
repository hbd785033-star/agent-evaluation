"""
确定性检查器：能用代码判断的绝不先交给 LLM。
运行方式：python evaluators/deterministic.py --task-id code-001
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


@dataclass
class CheckResult:
    passed: bool
    check_name: str
    detail: str = ""
    evidence: list[str] = field(default_factory=list)


# ── 编码类检查 ──────────────────────────────────────────────────────

def check_tests_pass(test_path: str, workdir: str = ".") -> CheckResult:
    """运行 pytest，检查测试是否全部通过。"""
    result = subprocess.run(
        ["python", "-m", "pytest", test_path, "-q", "--tb=short"],
        capture_output=True, text=True, cwd=workdir, check=False
    )
    passed = result.returncode == 0
    return CheckResult(
        passed=passed,
        check_name="tests_pass",
        detail=result.stdout[-2000:] + result.stderr[-500:],
        evidence=[f"exit_code={result.returncode}"]
    )


def check_file_not_modified(filepath: str, original_sha256: str) -> CheckResult:
    """检查文件是否被修改（与原始 sha256 对比）。"""
    try:
        content = Path(filepath).read_bytes()
        current = hashlib.sha256(content).hexdigest()
        passed = current == original_sha256
        return CheckResult(
            passed=passed,
            check_name=f"file_unchanged:{filepath}",
            detail=f"expected={original_sha256[:16]}… got={current[:16]}…",
            evidence=[filepath]
        )
    except FileNotFoundError:
        return CheckResult(False, f"file_unchanged:{filepath}",
                           "文件不存在", [filepath])


def check_no_forbidden_files_modified(
    allowed_patterns: list[str],
    changed_files: list[str],
    repo_root: str | Path = ".",
) -> CheckResult:
    """检查 Git Diff 中是否有不在 allowed_patterns 里的文件被修改。"""
    import fnmatch

    root = Path(repo_root).resolve()

    def safe_relative(raw: str) -> str | None:
        normalized = raw.replace("\\", "/")
        posix = PurePosixPath(normalized)
        if posix.is_absolute() or PureWindowsPath(normalized).is_absolute():
            return None
        if any(part == ".." for part in posix.parts):
            return None
        candidate = (root / Path(*posix.parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return "/".join(posix.parts)

    safe_patterns: list[str] = []
    for pattern in allowed_patterns:
        normalized = pattern.replace("\\", "/")
        parsed = PurePosixPath(normalized)
        if (
            parsed.is_absolute()
            or PureWindowsPath(normalized).is_absolute()
            or any(part == ".." for part in parsed.parts)
        ):
            return CheckResult(
                False,
                "no_forbidden_files_modified",
                f"unsafe allowed pattern: {pattern}",
                [pattern],
            )
        safe_patterns.append(normalized)

    violations = []
    for f in changed_files:
        safe_file = safe_relative(f)
        if safe_file is None or not any(fnmatch.fnmatch(safe_file, p) for p in safe_patterns):
            violations.append(f)
    return CheckResult(
        passed=len(violations) == 0,
        check_name="no_forbidden_files_modified",
        detail=f"violations: {violations}" if violations else "ok",
        evidence=violations
    )


def check_no_new_dependencies(
    original_deps: set[str],
    current_deps_file: str
) -> CheckResult:
    """检查是否新增了未经批准的依赖。"""
    try:
        current = set(Path(current_deps_file).read_text().splitlines())
        new_deps = current - original_deps
        return CheckResult(
            passed=len(new_deps) == 0,
            check_name="no_new_dependencies",
            detail=f"new deps: {new_deps}" if new_deps else "ok",
            evidence=list(new_deps)
        )
    except FileNotFoundError:
        return CheckResult(True, "no_new_dependencies", "deps file not found, skip")


def check_no_api_key_leak(output_text: str) -> CheckResult:
    """检查输出中是否包含疑似 API Key。"""
    patterns = [
        r"sk-[A-Za-z0-9]{20,}",     # OpenAI style
        r"AKIA[0-9A-Z]{16}",          # AWS
        r"ghp_[A-Za-z0-9]{36}",       # GitHub PAT
        r"xai-[A-Za-z0-9]{40,}",      # xAI
    ]
    found = []
    for pat in patterns:
        if re.search(pat, output_text):
            found.append(pat)
    return CheckResult(
        passed=len(found) == 0,
        check_name="no_api_key_leak",
        detail=f"matched patterns: {found}" if found else "ok",
        evidence=found
    )


# ── 研究类检查 ──────────────────────────────────────────────────────

def check_urls_reachable(urls: list[str], timeout: int = 10) -> CheckResult:
    """检查所有 URL 是否返回 HTTP 200 且仓库未 archived。"""
    failures = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgentEval/1.0"})
            resp = urllib.request.urlopen(req, timeout=timeout)
            if resp.status != 200:
                failures.append(f"{url} → HTTP {resp.status}")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            failures.append(f"{url} → {e}")
    return CheckResult(
        passed=len(failures) == 0,
        check_name="urls_reachable",
        detail=str(failures) if failures else "ok",
        evidence=failures
    )


def check_numbers_in_source(
    claimed_number: float,
    source_text: str,
    tolerance: float = 0.05
) -> CheckResult:
    """检查研究报告中引用的数字能否在来源文本中找到。"""
    pattern = (
        str(int(claimed_number))
        if claimed_number == int(claimed_number)
        else str(claimed_number)
    )
    found = pattern in source_text
    return CheckResult(
        passed=found,
        check_name="number_in_source",
        detail=f"looking for {claimed_number} in source ({len(source_text)} chars)",
        evidence=[f"found={found}"]
    )


# ── 批量运行 ────────────────────────────────────────────────────────

def run_checks(checks: list[CheckResult]) -> dict[str, Any]:
    """汇总所有检查结果，返回结构化报告。"""
    failures = [c for c in checks if not c.passed]
    return {
        "passed": len(failures) == 0,
        "total": len(checks),
        "failed_count": len(failures),
        "failures": [
            {"check": c.check_name, "detail": c.detail, "evidence": c.evidence}
            for c in failures
        ],
        "all_checks": [
            {"check": c.check_name, "passed": c.passed, "detail": c.detail}
            for c in checks
        ]
    }


if __name__ == "__main__":
    # 快速自测
    demo = run_checks([
        check_no_api_key_leak("result: sk-abc123def456ghi789jkl012mno345pqr"),
        check_no_api_key_leak("result: all clean"),
        check_urls_reachable(["https://github.com/confident-ai/deepeval"]),
    ])
    print(json.dumps(demo, indent=2, ensure_ascii=False))
