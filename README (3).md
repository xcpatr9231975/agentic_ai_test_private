# 🤖 Autonomous Coding Agent

A continuously running Python agent that **generates → reviews → tests → improves** code autonomously using local LLMs via [Ollama](https://ollama.com).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN LOOP (infinite)                  │
│                                                         │
│   ┌──────────┐   ┌──────────┐   ┌───────┐   ┌───────┐ │
│   │ GENERATE │──▶│  REVIEW  │──▶│ TEST  │──▶│FEATURE│─┐│
│   │  (LLM)   │   │  (LLM)   │   │(lint+ │   │ (LLM) │ ││
│   └──────────┘   └──────────┘   │ exec) │   └───────┘ ││
│        ▲                        └───┬───┘       │      ││
│        │                            │           │      ││
│        │          ┌─────────┐       │           │      ││
│        │          │ LLM FIX │◀──────┘           │      ││
│        │          │(on fail)│   (if errors)     │      ││
│        │          └─────────┘                   │      ││
│        └────────────────────────────────────────┘      ││
│                                                         │
│   workspace/                                            │
│   ├── solution.py          ← latest generated code      │
│   ├── state.json           ← agent state (resumable)    │
│   └── iteration_history.json ← full audit trail         │
└─────────────────────────────────────────────────────────┘
```

## Quickstart

### 1. Install Ollama & pull a model

```bash
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5-coder:32b    # recommended for coding
# or any model you prefer:
# ollama pull llama3:70b
# ollama pull codellama:34b
# ollama pull deepseek-coder-v2:16b
```

### 2. Install Python dependencies

```bash
pip install requests rich
```

### 3. Run the agent

```bash
# Basic usage — give it a task
python agent.py "Create a Python script that checks BGP adjacency between two routers using Netmiko"

# Use a specific model
python agent.py "Build an SNMP network scanner" --model llama3:70b

# Limit iterations
python agent.py "Create a ping sweep tool" --max-iter 5

# Resume a previous session
python agent.py --resume

# Custom workspace
python agent.py "Build a syslog parser" --workspace ./my_project
```

### 4. Watch it work

The agent will:
1. **Generate** initial code from your task description
2. **Review** the code for bugs, security issues, style problems
3. **Test** it (syntax check, linting, import test)
4. **Add a feature** to make it more robust/useful
5. **Loop** back to step 2 — continuously improving

Press `Ctrl+C` to stop at any time. The state is saved and you can `--resume`.

## Configuration

| Env Variable     | Default                    | Description                    |
|-----------------|----------------------------|--------------------------------|
| `OLLAMA_URL`    | `http://localhost:11434`   | Ollama server URL              |
| `OLLAMA_MODEL`  | `qwen2.5-coder:32b`       | Default model                  |
| `MAX_ITERATIONS`| `0` (infinite)             | Stop after N iterations        |
| `ITERATION_PAUSE`| `5`                       | Seconds between cycles         |

## CLI Options

```
positional arguments:
  task                    Task description for the agent

options:
  -m, --model MODEL       Ollama model name
  -r, --resume            Resume from saved state
  -n, --max-iter N        Max iterations (0=infinite)
  -p, --pause SECONDS     Seconds between iterations
  -w, --workspace DIR     Workspace directory
```

## Example Session

```
$ python agent.py "Create a BGP adjacency checker for Cisco routers using Netmiko"

╔══════════════════════════════════════════════════════╗
║ 🤖 Autonomous Coding Agent                          ║
║                                                      ║
║ Task:   Create a BGP adjacency checker ...           ║
║ Model:  qwen2.5-coder:32b                           ║
║ Workspace: /home/user/workspace                      ║
╚══════════════════════════════════════════════════════╝

─── Iteration 1 ───────────────────────────────────────
╔═ Phase 1: CODE GENERATION ═══════════════════════════╗
  → Querying qwen2.5-coder:32b... (24.3s)
  [generated 85 lines of code]

╔═ Phase 2: CODE REVIEW ═══════════════════════════════╗
  → Querying qwen2.5-coder:32b... (18.7s)
  ✓ Code updated from review

╔═ Phase 3: TESTING ═══════════════════════════════════╗
  ✓ Syntax OK
  ✓ flake8 passed
  ✓ Import test passed

╔═ Phase 4: FEATURE EXPANSION ═════════════════════════╗
  → Querying qwen2.5-coder:32b... (22.1s)
  ✓ Feature added: Added JSON output format support

─── Iteration 2 ───────────────────────────────────────
  ... (continues improving) ...
```

## How It Handles Failures

- **Syntax errors** → sends error to LLM for automatic fix
- **Lint warnings** → included in next review cycle
- **Import failures** → LLM receives traceback and fixes
- **LLM returns garbage** → keeps previous working version
- **Ollama down** → exits with clear error message
- **Ctrl+C** → saves state cleanly, can resume later

## Tips

- **Bigger models = better code**: 32B+ models produce significantly better results
- **First iteration is key**: give a detailed task description for better initial code
- **Check workspace/**: all code versions and history are saved
- **Use --max-iter for experiments**: try 3-5 iterations to see if it converges
- **Review iteration_history.json**: full audit trail of what the agent did and why
