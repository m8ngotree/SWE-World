"""
lang_utils.py — Language-aware utilities for multi-language SWE-World.

This module is the single source of truth for all language-specific logic:
  - Language enum
  - Per-language command patterns (replaces PYTHON_CMD_PATTERN)
  - Language detection from repo name or command string
  - Generalised is_execution_command()
  - Per-language test-output parsers (replaces the pytest-only extract_test_report)
  - SWT and SWR system prompts keyed by Language

All other modules import from here; this file has no internal project imports
so there are no circular-dependency risks.
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Language enum
# ---------------------------------------------------------------------------

class Language(Enum):
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    RUST = auto()
    GO = auto()
    JAVA = auto()


# ---------------------------------------------------------------------------
# Per-language command patterns
# ---------------------------------------------------------------------------

CMD_PATTERNS: Dict[Language, re.Pattern] = {
    Language.PYTHON: re.compile(
        r'\b(python\d*|pytest)\b'
    ),
    Language.JAVASCRIPT: re.compile(
        r'\b(jest|mocha|jasmine|vitest|'
        r'npm\s+(?:run\s+)?test|'
        r'yarn\s+(?:run\s+)?test|'
        r'npx\s+(?:jest|mocha|vitest))\b'
    ),
    Language.TYPESCRIPT: re.compile(
        r'\b(jest|ts-jest|mocha|vitest|'
        r'npm\s+(?:run\s+)?test|'
        r'yarn\s+(?:run\s+)?test|'
        r'npx\s+(?:jest|ts-jest|mocha|vitest))\b'
    ),
    Language.RUST: re.compile(
        r'\bcargo\s+(?:test|bench|nextest\s+run)\b'
    ),
    Language.GO: re.compile(
        r'\bgo\s+test\b'
    ),
    Language.JAVA: re.compile(
        r'\b(?:mvn|./mvnw)\s+(?:test|verify|surefire:test)|'
        r'(?:gradle|./gradlew)\s+(?:test|check)|'
        r'\bjunit\b'
    ),
}

# Package-manager install commands to skip (analogous to "pip install" skip for Python)
INSTALL_SKIP_PATTERNS: Dict[Language, List[str]] = {
    Language.PYTHON:     ["pip install", "pip3 install", "conda install"],
    Language.JAVASCRIPT: ["npm install", "npm ci", "yarn install", "yarn add", "pnpm install"],
    Language.TYPESCRIPT: ["npm install", "npm ci", "yarn install", "yarn add", "pnpm install"],
    Language.RUST:       ["cargo add", "cargo install"],
    Language.GO:         ["go get", "go mod download", "go install"],
    Language.JAVA:       ["mvn install", "gradle dependencies"],
}


# ---------------------------------------------------------------------------
# Repo-name → Language mapping
# ---------------------------------------------------------------------------
# Keys are Language values; values are substrings to look for in repo names.
# The SUPPORTED_REPOS list in __init__.py is the authoritative flat list;
# this dict is used for detection only.

REPO_LANGUAGE_MAP: Dict[Language, List[str]] = {
    Language.JAVASCRIPT: [
        "express", "lodash", "moment", "axios", "chalk", "mocha",
        "react", "vue", "jquery", "koa", "hapi", "fastify",
        "socket.io", "socket-io", "nodemon", "browserify",
    ],
    Language.TYPESCRIPT: [
        "typescript", "eslint", "prettier", "webpack", "jest",
        "angular", "nestjs", "typeorm", "ts-node", "inversify",
        "rxjs", "typestack",
    ],
    Language.RUST: [
        "rust", "cargo", "tokio", "serde", "actix", "hyper",
        "rustfmt", "clippy", "rocket", "diesel", "reqwest",
    ],
    Language.GO: [
        "golang", "gin", "gorm", "cobra", "gorilla", "hugo",
        "kubernetes", "docker", "etcd", "prometheus",
    ],
    Language.JAVA: [
        "spring", "maven", "gradle", "junit", "hibernate",
        "mockito", "jackson", "guava", "netty",
    ],
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(repo_name: str = "", command: str = "") -> Language:
    """
    Detect the programming language for a given repo / command.

    Priority:
      1. Repo-name substring match against REPO_LANGUAGE_MAP.
      2. Command-pattern match against CMD_PATTERNS.
      3. Default: Language.PYTHON (matches existing behaviour).

    Args:
        repo_name: e.g. "pallets/flask", "expressjs/express"
        command:   shell command string, e.g. "npm test" or "cargo test"
    """
    if repo_name:
        repo_lower = repo_name.lower()
        for lang, keywords in REPO_LANGUAGE_MAP.items():
            if any(kw in repo_lower for kw in keywords):
                return lang

    if command:
        cmd_stripped = command.strip()
        for lang, pattern in CMD_PATTERNS.items():
            if pattern.search(cmd_stripped):
                return lang

    return Language.PYTHON


# ---------------------------------------------------------------------------
# Generalised is_execution_command()
# ---------------------------------------------------------------------------

def is_execution_command(code: str, lang: Language = Language.PYTHON) -> bool:
    """
    Return True if *code* is a test/execution command that SWT should simulate.

    This is a language-aware replacement for the original
    ``is_python_execution_command``.  The logic mirrors the Python original
    but uses per-language patterns and skip lists.

    Rules (all must hold):
      1. ``code`` must be non-empty.
      2. Must start with ``execute_bash``.
      3. Must not contain a package-manager *install* command.
      4. Must not contain a bare ``git`` sub-command (``git ``).
      5. Must match the language's CMD_PATTERN.
    """
    if not code:
        return False

    stripped = code.strip()

    # Rule 2: must start with execute_bash
    if not stripped.startswith("execute_bash"):
        return False

    # Rule 4: skip git commands
    if "git " in stripped:
        return False

    # Rule 3: skip package-manager install
    lowered = stripped.lower()
    for skip_phrase in INSTALL_SKIP_PATTERNS.get(lang, []):
        if skip_phrase in lowered:
            return False

    # Rule 5: pattern match
    # For Python we keep the two fast-path string checks from the original.
    if lang == Language.PYTHON:
        if "python3 " in stripped or "python " in stripped:
            return True

    return bool(CMD_PATTERNS[lang].search(stripped))


# Backward-compatible alias
def is_python_execution_command(code: str) -> bool:
    return is_execution_command(code, lang=Language.PYTHON)


# ---------------------------------------------------------------------------
# Per-language test-output parsers
# ---------------------------------------------------------------------------

def _extract_test_report_python(log: str) -> Tuple[bool, str, str]:
    """
    Extract the pytest test report section from a full log string.
    Identical logic to the original extract_test_report() in
    1_collect_world_model_sft_data.py so existing behaviour is preserved.
    """
    timeout_marker = "The command took too long to execute"
    if timeout_marker in log:
        return False, "Timeout: command took too long to execute.", ""

    start_marker = "============================= test session starts =============================="
    lines = log.splitlines(keepends=True)
    start_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break

    if start_idx == -1:
        return False, "Pytest start marker not found.", ""

    end_idx = None

    end_output_marker = ">>>>> End Test Output"
    for j in range(len(lines) - 1, -1, -1):
        if end_output_marker in lines[j]:
            if j - 1 >= start_idx:
                end_idx = j - 1
                break

    if end_idx is None:
        git_checkout_marker = "+ git checkout "
        for j in range(len(lines) - 1, -1, -1):
            if git_checkout_marker in lines[j]:
                if j - 1 >= start_idx:
                    end_idx = j - 1
                    break

    if end_idx is None:
        equal_line_marker = "===="
        for j in range(len(lines) - 1, -1, -1):
            if equal_line_marker in lines[j]:
                if j >= start_idx:
                    end_idx = j
                    break

    if end_idx is None or end_idx < start_idx:
        return False, "No valid end marker found.", ""

    test_report = "".join(lines[start_idx:end_idx + 1])
    return True, "", test_report


def _extract_test_report_jest(log: str) -> Tuple[bool, str, str]:
    """
    Extract the Jest test report section from a full log string.

    Jest output structure (simplified):
        PASS src/foo.test.ts
        FAIL src/bar.test.ts
          ● describe block › test name
            Error: …
        Test Suites: 1 failed, 1 passed, 2 total
        Tests:       1 failed, 3 passed, 4 total
        Snapshots:   0 total
        Time:        2.456 s
    """
    timeout_marker = "The command took too long to execute"
    if timeout_marker in log:
        return False, "Timeout: command took too long to execute.", ""

    lines = log.splitlines(keepends=True)

    # Start: first line that looks like a Jest PASS/FAIL file-level result
    start_idx = None
    jest_file_pattern = re.compile(r'^\s*(PASS|FAIL)\s+\S')
    for i, line in enumerate(lines):
        if jest_file_pattern.match(line):
            start_idx = i
            break

    # Fallback: look for the summary block
    if start_idx is None:
        for i, line in enumerate(lines):
            if re.search(r'Test Suites:|Tests:\s+\d+', line):
                start_idx = max(0, i - 2)
                break

    if start_idx is None:
        return False, "Jest start marker not found.", ""

    # End: last line containing "Time:" or "Tests:" summary
    end_idx = None
    for j in range(len(lines) - 1, start_idx - 1, -1):
        if re.search(r'^\s*Time:\s+[\d.]+\s*s', lines[j]):
            end_idx = j
            break
    if end_idx is None:
        for j in range(len(lines) - 1, start_idx - 1, -1):
            if re.search(r'^\s*Tests:\s+', lines[j]):
                end_idx = j
                break
    if end_idx is None:
        for j in range(len(lines) - 1, start_idx - 1, -1):
            if re.search(r'^\s*Test Suites:\s+', lines[j]):
                end_idx = j
                break

    if end_idx is None or end_idx < start_idx:
        return False, "Jest end marker not found.", ""

    test_report = "".join(lines[start_idx:end_idx + 1])
    return True, "", test_report


def _extract_test_report_cargo(log: str) -> Tuple[bool, str, str]:
    """
    Extract the cargo test report section from a full log string.

    Cargo test output structure:
        running 5 tests
        test test_a ... ok
        test test_b ... FAILED
        …
        failures:
            test_b
        test result: FAILED. 4 passed; 1 failed; 0 ignored; …
    """
    timeout_marker = "The command took too long to execute"
    if timeout_marker in log:
        return False, "Timeout: command took too long to execute.", ""

    lines = log.splitlines(keepends=True)

    start_idx = None
    running_pattern = re.compile(r'^\s*running \d+ tests?')
    for i, line in enumerate(lines):
        if running_pattern.match(line):
            start_idx = i
            break

    if start_idx is None:
        return False, "cargo test 'running N tests' marker not found.", ""

    # End: last "test result:" line
    end_idx = None
    for j in range(len(lines) - 1, start_idx - 1, -1):
        if re.search(r'test result:', lines[j]):
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(lines) - 1

    test_report = "".join(lines[start_idx:end_idx + 1])
    return True, "", test_report


def _extract_test_report_go(log: str) -> Tuple[bool, str, str]:
    """
    Extract the go test report section from a full log string.

    go test output structure:
        === RUN   TestFoo
        --- PASS: TestFoo (0.00s)
        --- FAIL: TestBar (0.01s)
            bar_test.go:15: assertion failed
        FAIL    github.com/user/repo    0.012s
        ok      github.com/user/repo    0.012s
    """
    timeout_marker = "The command took too long to execute"
    if timeout_marker in log:
        return False, "Timeout: command took too long to execute.", ""

    lines = log.splitlines(keepends=True)

    start_idx = None
    go_start_pattern = re.compile(r'^\s*(=== RUN|--- (?:PASS|FAIL):)')
    for i, line in enumerate(lines):
        if go_start_pattern.match(line):
            start_idx = i
            break

    # Fallback: "ok" or "FAIL" package summary
    if start_idx is None:
        for i, line in enumerate(lines):
            if re.match(r'^(ok|FAIL)\s+\S+', line):
                start_idx = i
                break

    if start_idx is None:
        return False, "go test start marker not found.", ""

    # End: last "ok <pkg>" or "FAIL <pkg>" summary line
    end_idx = None
    for j in range(len(lines) - 1, start_idx - 1, -1):
        if re.match(r'^(ok|FAIL)\s+\S+', lines[j]):
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(lines) - 1

    test_report = "".join(lines[start_idx:end_idx + 1])
    return True, "", test_report


def _extract_test_report_java(log: str) -> Tuple[bool, str, str]:
    """
    Extract the Maven/Gradle/JUnit test report from a log string.

    Maven surefire output structure:
        -------------------------------------------------------
         T E S T S
        -------------------------------------------------------
        Running com.example.FooTest
        Tests run: 4, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.123 s
        …
        BUILD FAILURE / BUILD SUCCESS
    """
    timeout_marker = "The command took too long to execute"
    if timeout_marker in log:
        return False, "Timeout: command took too long to execute.", ""

    lines = log.splitlines(keepends=True)

    # Maven: look for "T E S T S" block
    start_idx = None
    for i, line in enumerate(lines):
        if "T E S T S" in line or re.search(r'Running \S+Test', line):
            start_idx = i
            break

    # Gradle: look for test result summary
    if start_idx is None:
        for i, line in enumerate(lines):
            if re.search(r'\d+ tests? completed', line):
                start_idx = max(0, i - 5)
                break

    if start_idx is None:
        return False, "Java/Maven/Gradle test start marker not found.", ""

    # End: BUILD SUCCESS/FAILURE or "tests completed"
    end_idx = None
    for j in range(len(lines) - 1, start_idx - 1, -1):
        if re.search(r'BUILD (SUCCESS|FAILURE)|tests? completed', lines[j]):
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(lines) - 1

    test_report = "".join(lines[start_idx:end_idx + 1])
    return True, "", test_report


# Dispatch table
_REPORT_EXTRACTORS = {
    Language.PYTHON:     _extract_test_report_python,
    Language.JAVASCRIPT: _extract_test_report_jest,
    Language.TYPESCRIPT: _extract_test_report_jest,   # same runner
    Language.RUST:       _extract_test_report_cargo,
    Language.GO:         _extract_test_report_go,
    Language.JAVA:       _extract_test_report_java,
}


def extract_test_report(log: str, lang: Language = Language.PYTHON) -> Tuple[bool, str, str]:
    """
    Language-aware test report extractor.

    Returns:
        (success, reason, test_report)
        success     — whether extraction succeeded
        reason      — failure reason string (empty on success)
        test_report — the extracted report text
    """
    extractor = _REPORT_EXTRACTORS.get(lang, _extract_test_report_python)
    return extractor(log)


# ---------------------------------------------------------------------------
# SWT (Transition Model) system prompts — keyed by Language
# ---------------------------------------------------------------------------

_SWT_PYTHON = """\
You are an expert Python code execution simulator and a world-class software engineer.
Your task is to predict the output of a given Python command within a specific code \
repository context.
Analyze all the provided information: the initial analysis, the problem description, \
human hints, the agent's current changes, the ideal "gold" solution, and the original \
content of the modified files.

Your output MUST be a single JSON object containing 'stdout', 'stderr', and 'exit_code'. \
Do not add any explanations or text outside of this JSON block.

### Key Information You Must Use:
1. **Initial Analysis of the Problem**: Core analysis of the issue — error behaviour, \
the bug, how it manifests, and the intended fix.
2. **Problem Description**: The specific issue the agent is trying to resolve.
3. **Command to Simulate**: The exact shell command to run.
4. **Content of Code to be Executed**: The actual source files being executed.
5. **Agent's Current Code Modifications (Patch)**: The diff the agent applied.
6. **Gold Standard Patch (For Your Reference)**: The correct fix for comparison.

### Your Task:
- Simulate realistic pytest / python output for the given command.
- For pytest commands: report which FAIL_TO_PASS / PASS_TO_PASS tests pass or fail.
- For reproduce-issue scripts: simulate the error described in the problem statement \
  if the patch has not yet fixed it, or clean output if it has.
- Do NOT fabricate test cases not mentioned in the context.

### Output Format:
{"stdout": "...", "stderr": "...", "exit_code": 0_or_nonzero}
"""

_SWT_JAVASCRIPT = """\
You are an expert JavaScript/Node.js code execution simulator and a world-class \
software engineer.
Your task is to predict the output of a given Jest / Mocha / npm-test command within \
a specific JavaScript or TypeScript repository.

Your output MUST be a single JSON object with 'stdout', 'stderr', and 'exit_code'.
Do NOT include any text outside that JSON block.

### Jest output format you must replicate:
  PASS  src/foo.test.js (1.234 s)
  FAIL  src/bar.test.js
    ● describe block › test name
      Error: <message>
        at Object.<anonymous> (src/bar.test.js:15:5)
  Test Suites: 1 failed, 1 passed, 2 total
  Tests:       1 failed, 3 passed, 4 total
  Snapshots:   0 total
  Time:        1.234 s

### Key Information You Must Use:
1. **Initial Analysis**: Core bug, symptoms, intended fix.
2. **Problem Description**: The issue being resolved.
3. **Command to Simulate**: The exact npm/jest/mocha invocation.
4. **Content of Code to be Executed**: The relevant source / test files.
5. **Agent's Current Patch**: The diff applied by the agent.
6. **Gold Standard Patch**: The correct fix for reference.
7. **FAIL_TO_PASS**: Tests that must now pass if the fix is correct.
8. **PASS_TO_PASS**: Regression tests that must continue to pass.

### Output Format:
{"stdout": "...", "stderr": "...", "exit_code": 0_or_nonzero}
"""

_SWT_TYPESCRIPT = _SWT_JAVASCRIPT.replace(
    "JavaScript/Node.js", "TypeScript/Node.js"
).replace(
    "JavaScript or TypeScript", "TypeScript or JavaScript"
)

_SWT_RUST = """\
You are an expert Rust code execution simulator and a world-class software engineer.
Your task is to predict the output of a given `cargo test` command within a specific \
Rust repository context.

Your output MUST be a single JSON object with 'stdout', 'stderr', and 'exit_code'.

### cargo test output format you must replicate:
  running 5 tests
  test module::test_a ... ok
  test module::test_b ... FAILED
  test module::test_c ... ok

  failures:
      module::test_b

  test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; \
finished in 0.01s

### Key Information You Must Use:
1. **Initial Analysis**: Core bug, symptoms, intended fix.
2. **Problem Description**: The issue being resolved.
3. **Command to Simulate**: The exact cargo invocation.
4. **Content of Code to be Executed**: Relevant source / test files.
5. **Agent's Current Patch**: The diff applied.
6. **Gold Standard Patch**: The correct fix.
7. **FAIL_TO_PASS** / **PASS_TO_PASS**: Test names that must pass / keep passing.

### Output Format:
{"stdout": "...", "stderr": "...", "exit_code": 0_or_nonzero}
"""

_SWT_GO = """\
You are an expert Go code execution simulator and a world-class software engineer.
Your task is to predict the output of a given `go test` command within a specific \
Go repository context.

Your output MUST be a single JSON object with 'stdout', 'stderr', and 'exit_code'.

### go test output format you must replicate:
  === RUN   TestFoo
  --- PASS: TestFoo (0.00s)
  === RUN   TestBar
      bar_test.go:15: assertion failed: got 1, want 2
  --- FAIL: TestBar (0.00s)
  FAIL    github.com/user/repo    0.012s
  ok      github.com/user/repo    0.001s  (when all pass)

### Key Information You Must Use:
1. **Initial Analysis**: Core bug, symptoms, intended fix.
2. **Problem Description**: The issue being resolved.
3. **Command to Simulate**: The exact go test invocation.
4. **Content of Code to be Executed**: Relevant source / test files.
5. **Agent's Current Patch**: The diff applied.
6. **Gold Standard Patch**: The correct fix.
7. **FAIL_TO_PASS** / **PASS_TO_PASS**: Test names.

### Output Format:
{"stdout": "...", "stderr": "...", "exit_code": 0_or_nonzero}
"""

_SWT_JAVA = """\
You are an expert Java code execution simulator and a world-class software engineer.
Your task is to predict the output of a given Maven / Gradle test command within a \
specific Java repository context.

Your output MUST be a single JSON object with 'stdout', 'stderr', and 'exit_code'.

### Maven surefire output format you must replicate:
  -------------------------------------------------------
   T E S T S
  -------------------------------------------------------
  Running com.example.FooTest
  Tests run: 4, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.123 s <<< FAILURE!

  Results:
  Tests in error:
    testBaz(com.example.FooTest): expected <1> but was <2>

  Tests run: 4, Failures: 0, Errors: 1, Skipped: 0

  BUILD FAILURE

### Key Information You Must Use:
1. **Initial Analysis**: Core bug, symptoms, intended fix.
2. **Problem Description**: The issue being resolved.
3. **Command to Simulate**: The Maven/Gradle invocation.
4. **Content of Code to be Executed**: Relevant source / test files.
5. **Agent's Current Patch**: The diff applied.
6. **Gold Standard Patch**: The correct fix.
7. **FAIL_TO_PASS** / **PASS_TO_PASS**: Test names.

### Output Format:
{"stdout": "...", "stderr": "...", "exit_code": 0_or_nonzero}
"""

SWT_SYSTEM_PROMPTS: Dict[Language, str] = {
    Language.PYTHON:     _SWT_PYTHON,
    Language.JAVASCRIPT: _SWT_JAVASCRIPT,
    Language.TYPESCRIPT: _SWT_TYPESCRIPT,
    Language.RUST:       _SWT_RUST,
    Language.GO:         _SWT_GO,
    Language.JAVA:       _SWT_JAVA,
}


# ---------------------------------------------------------------------------
# SWR (Reward Model) system prompts — keyed by Language
# ---------------------------------------------------------------------------

_SWR_BASE = """\
### Reward Rules (apply to ALL languages):
- reward = 1  iff  ALL FAIL_TO_PASS tests pass  AND  ALL PASS_TO_PASS tests pass.
- reward = 0  if   ANY FAIL_TO_PASS test fails or errors.
- reward = 0  if   ANY PASS_TO_PASS test fails or errors.

### Output Format
Your answer MUST include:
1. A `<think>…</think>` block with your full reasoning.
2. Immediately after, a single JSON object:
   {"test_report": "…concise summary…", "reward": 0_or_1}
No other text outside these two sections.
"""

_SWR_PYTHON = f"""\
You are an expert software engineering test runner and evaluator for Python projects.
Your task is to simulate running a pytest command inside a Python repository, then:
1. Produce a realistic pytest test report (use pytest output format).
2. Decide a final binary reward.

### pytest test report format:
  ============================= test session starts ==============================
  collected N items
  tests/test_foo.py::test_bar PASSED
  tests/test_foo.py::test_baz FAILED
  …
  ========================= short test summary info ==========================
  FAILED tests/test_foo.py::test_baz - AssertionError: …
  ========================= N passed, M failed in X.XXs =========================

{_SWR_BASE}
"""

_SWR_JAVASCRIPT = f"""\
You are an expert software engineering test runner and evaluator for JavaScript/TypeScript \
projects using Jest or Mocha.
Your task is to simulate running a Jest / npm-test command, then:
1. Produce a realistic Jest test report (use Jest's stdout format).
2. Decide a final binary reward.

### Jest test report format:
  PASS  src/utils.test.js (0.543 s)
  FAIL  src/core.test.js
    ● describe › test name
      Error: expected 1 but got 2
        at Object.<anonymous> (src/core.test.js:22:5)
  Test Suites: 1 failed, 1 passed, 2 total
  Tests:       1 failed, 3 passed, 4 total
  Time:        0.987 s

Test IDs in FAIL_TO_PASS / PASS_TO_PASS follow the pattern:
  "describe block > test name"  or  "test name"  (top-level)

{_SWR_BASE}
"""

_SWR_TYPESCRIPT = _SWR_JAVASCRIPT.replace(
    "JavaScript/TypeScript", "TypeScript/JavaScript"
)

_SWR_RUST = f"""\
You are an expert software engineering test runner and evaluator for Rust projects.
Your task is to simulate running `cargo test`, then:
1. Produce a realistic cargo test report.
2. Decide a final binary reward.

### cargo test report format:
  running 5 tests
  test module::test_passes ... ok
  test module::test_fails  ... FAILED
  …
  failures:
      module::test_fails

  test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out

Test IDs in FAIL_TO_PASS / PASS_TO_PASS follow the pattern:
  "module::test_function_name"

{_SWR_BASE}
"""

_SWR_GO = f"""\
You are an expert software engineering test runner and evaluator for Go projects.
Your task is to simulate running `go test`, then:
1. Produce a realistic go test report.
2. Decide a final binary reward.

### go test report format:
  === RUN   TestFoo
  --- PASS: TestFoo (0.00s)
  === RUN   TestBar
      bar_test.go:15: expected 1, got 2
  --- FAIL: TestBar (0.00s)
  FAIL    github.com/user/repo    0.015s

  or, on full success:
  ok      github.com/user/repo    0.001s

Test IDs in FAIL_TO_PASS / PASS_TO_PASS follow the pattern:
  "TestFunctionName"  or  "TestSuite/SubTest"

{_SWR_BASE}
"""

_SWR_JAVA = f"""\
You are an expert software engineering test runner and evaluator for Java projects \
using Maven Surefire or Gradle.
Your task is to simulate running `mvn test` or `gradle test`, then:
1. Produce a realistic Maven/Gradle test report.
2. Decide a final binary reward.

### Maven surefire format:
  -------------------------------------------------------
   T E S T S
  -------------------------------------------------------
  Running com.example.FooTest
  Tests run: 3, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.1 s <<< FAILURE!
  …
  BUILD FAILURE / BUILD SUCCESS

Test IDs in FAIL_TO_PASS / PASS_TO_PASS follow the pattern:
  "com.example.FooTest#testMethodName"  or  "com.example.FooTest"

{_SWR_BASE}
"""

SWR_SYSTEM_PROMPTS: Dict[Language, str] = {
    Language.PYTHON:     _SWR_PYTHON,
    Language.JAVASCRIPT: _SWR_JAVASCRIPT,
    Language.TYPESCRIPT: _SWR_TYPESCRIPT,
    Language.RUST:       _SWR_RUST,
    Language.GO:         _SWR_GO,
    Language.JAVA:       _SWR_JAVA,
}
