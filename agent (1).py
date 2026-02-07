#!/usr/bin/env python3
"""
Autonomous Coding Agent
=======================
A continuously running agent that:
  1. Generates code from a task description (using Ollama LLM)
  2. Reviews the generated code for bugs, style, security
  3. Runs tests (lint + execution)
  4. Proposes & implements new features or fixes
  5. Loops back to step 2 — forever improving

Requires: ollama running locally with a model pulled (e.g. qwen2.5-coder:32b)
Install deps:  pip install requests rich
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import textwrap
import argparse
import datetime
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("Missing 'requests'. Install with:  pip install requests")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import box
except ImportError:
    sys.exit("Missing 'rich'. Install with:  pip install rich")

# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
WORKSPACE_DIR = Path("workspace")
HISTORY_FILE = WORKSPACE_DIR / "iteration_history.json"
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "0"))  # 0 = infinite
ITERATION_PAUSE = int(os.getenv("ITERATION_PAUSE", "5"))  # seconds between cycles

console = Console()

# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    iteration: int
    phase: str  # generate | review | test | feature
    timestamp: str
    model: str
    code_version: str
    review_notes: str = ""
    test_output: str = ""
    lint_output: str = ""
    feature_added: str = ""
    success: bool = True
    error: str = ""


@dataclass
class AgentState:
    task_description: str
    current_code: str = ""
    iteration: int = 0
    features_added: list = field(default_factory=list)
    history: list = field(default_factory=list)
    filename: str = "solution.py"

    def save(self):
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "task_description": self.task_description,
            "current_code": self.current_code,
            "iteration": self.iteration,
            "features_added": self.features_added,
            "filename": self.filename,
        }
        (WORKSPACE_DIR / "state.json").write_text(json.dumps(data, indent=2))
        # Always write latest code to file
        (WORKSPACE_DIR / self.filename).write_text(self.current_code)

    @classmethod
    def load(cls) -> Optional["AgentState"]:
        state_file = WORKSPACE_DIR / "state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return cls(**{k: v for k, v in data.items() if k != "history"})
        return None


# ─── Ollama LLM Interface ───────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                console.print(
                    f"[yellow]⚠ Model '{self.model}' not found. "
                    f"Available: {', '.join(models) or 'none'}[/yellow]"
                )
                console.print(
                    f"[yellow]  Pull it with: ollama pull {self.model}[/yellow]"
                )
        except requests.ConnectionError:
            console.print(
                f"[red]✗ Cannot connect to Ollama at {self.base_url}\n"
                f"  Start it with: ollama serve[/red]"
            )
            sys.exit(1)

    def generate(self, prompt: str, system: str = "", temperature: float = 0.4) -> str:
        """Send a prompt to Ollama and return the full response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
            },
        }
        try:
            console.print(f"  [dim]→ Querying {self.model}...[/dim]", end="")
            t0 = time.time()
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            console.print(f" [dim]({elapsed:.1f}s)[/dim]")
            return r.json().get("response", "").strip()
        except Exception as e:
            console.print(f"\n[red]  ✗ LLM error: {e}[/red]")
            return ""


# ─── Code Extraction Helper ─────────────────────────────────────────────────

def extract_python_code(text: str) -> str:
    """Extract the largest Python code block from LLM response."""
    # Try fenced code blocks first
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # Fallback: look for lines that look like Python
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#!", "# ")) or in_code:
            code_lines.append(line)
            in_code = True
        elif in_code and (line.startswith(" ") or line.startswith("\t") or stripped == ""):
            code_lines.append(line)
        elif in_code and stripped and not line.startswith(" "):
            # Check if this could be top-level python
            if any(stripped.startswith(k) for k in (
                "if ", "for ", "while ", "try:", "except", "with ",
                "print(", "return ", "raise ", "assert "
            )):
                code_lines.append(line)
            else:
                in_code = False
    return "\n".join(code_lines).strip() if code_lines else text.strip()


# ─── Agent Phases ────────────────────────────────────────────────────────────

class AutonomousAgent:
    def __init__(self, llm: OllamaClient, state: AgentState):
        self.llm = llm
        self.state = state
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Generate Initial Code ───────────────────────────────────
    def phase_generate(self) -> IterationResult:
        console.print(Panel(
            "[bold cyan]Phase 1: CODE GENERATION[/bold cyan]",
            box=box.DOUBLE,
        ))

        system_prompt = textwrap.dedent("""\
            You are an expert Python network programmer.
            Write clean, production-quality Python code.
            Include proper error handling, logging, type hints, and docstrings.
            Use standard library where possible; if external libs are needed, note them.
            Return ONLY the Python code inside a ```python code block.
        """)

        user_prompt = f"""\
Task: {self.state.task_description}

Requirements:
- Well-structured with functions/classes
- Proper error handling (try/except)
- Logging via the `logging` module
- Type hints on all functions
- A `main()` entry point with argparse if CLI args are needed
- Docstrings on all public functions/classes

Write the complete Python script now.
"""
        response = self.llm.generate(user_prompt, system=system_prompt, temperature=0.3)
        code = extract_python_code(response)

        if not code:
            return IterationResult(
                iteration=self.state.iteration, phase="generate",
                timestamp=_now(), model=self.llm.model,
                code_version="", success=False, error="LLM returned no code",
            )

        self.state.current_code = code
        self.state.save()

        console.print(Syntax(code, "python", line_numbers=True, theme="monokai"))

        return IterationResult(
            iteration=self.state.iteration, phase="generate",
            timestamp=_now(), model=self.llm.model,
            code_version=code[:200] + "...",
        )

    # ── Phase 2: Review Code ─────────────────────────────────────────────
    def phase_review(self) -> IterationResult:
        console.print(Panel(
            "[bold yellow]Phase 2: CODE REVIEW[/bold yellow]",
            box=box.DOUBLE,
        ))

        system_prompt = textwrap.dedent("""\
            You are a senior Python code reviewer specializing in network automation.
            Review the code for: bugs, security issues, error handling gaps,
            performance problems, style issues, missing edge cases.
            Be specific and actionable. Categorize issues as:
            [CRITICAL] - Must fix, will cause failures
            [WARNING]  - Should fix, potential problems
            [INFO]     - Nice to have improvements
            After listing issues, provide the COMPLETE corrected code in a ```python block.
        """)

        user_prompt = f"""\
Original task: {self.state.task_description}

Code to review:
```python
{self.state.current_code}
```

Please review this code thoroughly and provide the improved version.
"""
        response = self.llm.generate(user_prompt, system=system_prompt, temperature=0.2)

        # Extract review notes (everything before the code block)
        review_notes = response.split("```")[0].strip() if "```" in response else response[:500]
        improved_code = extract_python_code(response)

        if improved_code and len(improved_code) > 50:
            self.state.current_code = improved_code
            self.state.save()
            console.print("[green]  ✓ Code updated from review[/green]")
        else:
            console.print("[yellow]  ⚠ No improved code extracted, keeping current version[/yellow]")

        console.print(Panel(Markdown(review_notes[:1000]), title="Review Notes"))

        return IterationResult(
            iteration=self.state.iteration, phase="review",
            timestamp=_now(), model=self.llm.model,
            code_version=self.state.current_code[:200] + "...",
            review_notes=review_notes[:1000],
        )

    # ── Phase 3: Test Code ───────────────────────────────────────────────
    def phase_test(self) -> IterationResult:
        console.print(Panel(
            "[bold magenta]Phase 3: TESTING[/bold magenta]",
            box=box.DOUBLE,
        ))

        code_path = WORKSPACE_DIR / self.state.filename
        lint_output = ""
        test_output = ""
        success = True

        # 3a. Syntax check
        console.print("  [dim]Running syntax check...[/dim]")
        try:
            compile(self.state.current_code, self.state.filename, "exec")
            console.print("  [green]✓ Syntax OK[/green]")
        except SyntaxError as e:
            lint_output += f"SYNTAX ERROR: {e}\n"
            success = False
            console.print(f"  [red]✗ Syntax Error: {e}[/red]")

        # 3b. Pylint / flake8 (if available)
        for linter in ["flake8", "pylint"]:
            try:
                result = subprocess.run(
                    [linter, str(code_path), "--max-line-length=120"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.stdout.strip():
                    lint_output += f"\n--- {linter} ---\n{result.stdout[:1500]}"
                    console.print(f"  [yellow]{linter} found issues[/yellow]")
                else:
                    console.print(f"  [green]✓ {linter} passed[/green]")
            except FileNotFoundError:
                console.print(f"  [dim]{linter} not installed, skipping[/dim]")
            except subprocess.TimeoutExpired:
                pass

        # 3c. Dry-run with --help or quick import test
        console.print("  [dim]Running import test...[/dim]")
        test_script = textwrap.dedent(f"""\
            import sys, importlib.util
            spec = importlib.util.spec_from_file_location("solution", "{code_path}")
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                print("IMPORT_OK")
            except SystemExit:
                print("IMPORT_OK (sys.exit called)")
            except Exception as e:
                print(f"IMPORT_FAIL: {{type(e).__name__}}: {{e}}")
                sys.exit(1)
        """)
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            test_output = (result.stdout + result.stderr).strip()
            if "IMPORT_FAIL" in test_output:
                success = False
                console.print(f"  [red]✗ Import failed: {test_output}[/red]")
            else:
                console.print(f"  [green]✓ Import test passed[/green]")
        except subprocess.TimeoutExpired:
            test_output = "Import test timed out (possible infinite loop at module level)"
            success = False

        # 3d. If tests failed, ask LLM to fix
        if not success:
            console.print("  [yellow]→ Asking LLM to fix issues...[/yellow]")
            fix_prompt = f"""\
The following code has issues that need fixing:

```python
{self.state.current_code}
```

Errors found:
{lint_output}
{test_output}

Fix ALL the issues and return the COMPLETE corrected code in a ```python block.
"""
            response = self.llm.generate(
                fix_prompt,
                system="You are an expert Python debugger. Fix the code and return the complete corrected version.",
                temperature=0.2,
            )
            fixed_code = extract_python_code(response)
            if fixed_code and len(fixed_code) > 50:
                self.state.current_code = fixed_code
                self.state.save()
                console.print("  [green]✓ Applied LLM fix[/green]")

        return IterationResult(
            iteration=self.state.iteration, phase="test",
            timestamp=_now(), model=self.llm.model,
            code_version=self.state.current_code[:200] + "...",
            lint_output=lint_output[:500],
            test_output=test_output[:500],
            success=success,
        )

    # ── Phase 4: Add Features ────────────────────────────────────────────
    def phase_add_feature(self) -> IterationResult:
        console.print(Panel(
            "[bold green]Phase 4: FEATURE EXPANSION[/bold green]",
            box=box.DOUBLE,
        ))

        already_added = "\n".join(f"  - {f}" for f in self.state.features_added) or "  (none yet)"

        system_prompt = textwrap.dedent("""\
            You are an expert Python developer who continuously improves code.
            Given a working script and its purpose, propose ONE meaningful new feature
            that makes the tool more useful, robust, or production-ready.
            Focus on practical improvements like:
            - Better error handling or retry logic
            - Configuration file support
            - Output formatting (JSON, CSV, table)
            - Logging improvements
            - Performance optimization
            - Additional protocol/network checks
            - Unit tests
            - Metrics or monitoring hooks
            First describe the feature in 1-2 sentences, then provide the COMPLETE
            updated code with the feature integrated, inside a ```python block.
        """)

        user_prompt = f"""\
Task: {self.state.task_description}

Features already added:
{already_added}

Current code:
```python
{self.state.current_code}
```

Propose and implement ONE new feature. Do NOT repeat already-added features.
Return the complete updated code.
"""
        response = self.llm.generate(user_prompt, system=system_prompt, temperature=0.5)

        # Extract feature description
        feature_desc = ""
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("```") and not line.startswith("#"):
                feature_desc = line[:200]
                break

        new_code = extract_python_code(response)
        if new_code and len(new_code) > len(self.state.current_code) * 0.5:
            self.state.current_code = new_code
            self.state.features_added.append(feature_desc or f"Iteration {self.state.iteration} improvement")
            self.state.save()
            console.print(f"  [green]✓ Feature added: {feature_desc}[/green]")
        else:
            console.print("  [yellow]⚠ Could not extract feature code, skipping[/yellow]")
            feature_desc = "(extraction failed)"

        return IterationResult(
            iteration=self.state.iteration, phase="feature",
            timestamp=_now(), model=self.llm.model,
            code_version=self.state.current_code[:200] + "...",
            feature_added=feature_desc,
        )

    # ── Main Loop ────────────────────────────────────────────────────────
    def run(self):
        console.print(Panel(
            f"[bold white]🤖 Autonomous Coding Agent[/bold white]\n\n"
            f"[cyan]Task:[/cyan] {self.state.task_description}\n"
            f"[cyan]Model:[/cyan] {self.llm.model}\n"
            f"[cyan]Workspace:[/cyan] {WORKSPACE_DIR.resolve()}\n"
            f"[cyan]Max iterations:[/cyan] {MAX_ITERATIONS or '∞'}",
            title="[bold]Starting Agent[/bold]",
            box=box.HEAVY,
        ))

        try:
            while True:
                self.state.iteration += 1
                iteration = self.state.iteration

                if MAX_ITERATIONS and iteration > MAX_ITERATIONS:
                    console.print(f"\n[bold]Reached max iterations ({MAX_ITERATIONS}). Stopping.[/bold]")
                    break

                console.rule(f"[bold] Iteration {iteration} ", style="bright_blue")

                # Phase 1: Generate (only first iteration)
                if iteration == 1 and not self.state.current_code:
                    result = self.phase_generate()
                    self._log_result(result)
                    if not result.success:
                        console.print("[red]Generation failed. Retrying next iteration...[/red]")
                        time.sleep(ITERATION_PAUSE)
                        continue

                # Phase 2: Review
                result = self.phase_review()
                self._log_result(result)

                # Phase 3: Test
                result = self.phase_test()
                self._log_result(result)

                # Phase 4: Add feature
                result = self.phase_add_feature()
                self._log_result(result)

                # Summary table
                self._print_summary()

                # Pause before next cycle
                console.print(f"\n[dim]Pausing {ITERATION_PAUSE}s before next iteration... (Ctrl+C to stop)[/dim]")
                time.sleep(ITERATION_PAUSE)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]⏹ Agent stopped by user[/bold yellow]")
        finally:
            self.state.save()
            self._print_final_report()

    def _log_result(self, result: IterationResult):
        self.state.history.append(asdict(result))
        # Persist history
        try:
            history = []
            if HISTORY_FILE.exists():
                history = json.loads(HISTORY_FILE.read_text())
            history.append(asdict(result))
            HISTORY_FILE.write_text(json.dumps(history, indent=2))
        except Exception:
            pass

    def _print_summary(self):
        table = Table(title=f"Iteration {self.state.iteration} Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Code lines", str(len(self.state.current_code.splitlines())))
        table.add_row("Features added", str(len(self.state.features_added)))
        table.add_row("Total iterations", str(self.state.iteration))
        table.add_row("Latest feature", self.state.features_added[-1] if self.state.features_added else "N/A")
        console.print(table)

    def _print_final_report(self):
        console.print(Panel(
            f"[bold]Final Report[/bold]\n\n"
            f"Iterations completed: {self.state.iteration}\n"
            f"Code file: {WORKSPACE_DIR.resolve() / self.state.filename}\n"
            f"Code lines: {len(self.state.current_code.splitlines())}\n"
            f"Features added: {len(self.state.features_added)}\n\n"
            + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(self.state.features_added)),
            title="[bold green]Agent Summary[/bold green]",
            box=box.HEAVY,
        ))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now().isoformat()


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Coding Agent — continuously writes, reviews, tests & improves code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s "Create a BGP adjacency checker between two routers using Netmiko"
              %(prog)s "Build a network device inventory scanner using SNMP" --model llama3:70b
              %(prog)s --resume   # Continue from last session

            Environment variables:
              OLLAMA_URL      Ollama server URL (default: http://localhost:11434)
              OLLAMA_MODEL    Default model (default: qwen2.5-coder:32b)
              MAX_ITERATIONS  Stop after N iterations (default: 0 = infinite)
              ITERATION_PAUSE Seconds between iterations (default: 5)
        """),
    )
    parser.add_argument("task", nargs="?", help="Task description for the agent")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from saved state")
    parser.add_argument("--max-iter", "-n", type=int, default=0, help="Max iterations (0=infinite)")
    parser.add_argument("--pause", "-p", type=int, default=5, help="Seconds between iterations")
    parser.add_argument("--workspace", "-w", default="workspace", help="Workspace directory")

    args = parser.parse_args()

    global WORKSPACE_DIR, HISTORY_FILE, MAX_ITERATIONS, ITERATION_PAUSE
    WORKSPACE_DIR = Path(args.workspace)
    HISTORY_FILE = WORKSPACE_DIR / "iteration_history.json"
    MAX_ITERATIONS = args.max_iter
    ITERATION_PAUSE = args.pause

    # Resume or new task
    state = None
    if args.resume:
        state = AgentState.load()
        if state:
            console.print(f"[green]✓ Resumed from iteration {state.iteration}[/green]")
        else:
            console.print("[yellow]No saved state found.[/yellow]")

    if state is None:
        if not args.task:
            console.print("[red]Provide a task description or use --resume[/red]")
            parser.print_help()
            sys.exit(1)
        state = AgentState(task_description=args.task)

    llm = OllamaClient(model=args.model)
    agent = AutonomousAgent(llm=llm, state=state)
    agent.run()


if __name__ == "__main__":
    main()
