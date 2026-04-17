# Stride — Real-Time Frame-Based Agent

## Concept

Stride is a real-time agent that operates on a fixed-frame loop (1 minute/frame), inspired by autonomous driving. The agent continuously perceives, decides, and acts — the cognitive layer is the "road" that determines trajectory, not a driver waiting to be called.

Unlike traditional session-based agents that wait for input → produce output, Stride runs a continuous loop:
- The world runs continuously (like a car driving)
- The AI cognitive layer provides the "road" (thinking/planning) not the "engine" (execution)
- Each frame: sense → think → act → repeat

## Architecture

### Frame Loop (1 minute = 60s)

| Phase | Duration | Role |
|-------|----------|-------|
| Input Collection | 20s | Gather GUI state, CLI events, system sensors |
| Analysis & Decision | 20s | Process inputs, decide on actions |
| Output Execution | 20s | Execute decided actions, update world model |

### Pipeline Model

Multiple independent pipelines (inspired by brain/spinal cord analogy):

- **Fast Pipeline (Spinal)** — Real-time reflexes, anomaly detection, immediate reactions
- **Medium Pipeline (Brain)** — Current frame decisions, situational awareness
- **Slow Pipeline (Cortex)** — Multi-frame planning, long-term strategy

All pipelines share a **World Model** (state snapshot written to JSON).

### Sensors (Input Sources)

- **GUI** — Screen capture via `screencapture` + element structure
- **CLI/Chat** — Command input, chat messages
- **System State** — CPU, memory, disk via `top`/`df`
- **Future** — Webcam (eyes), microphone, etc.

### Outputs (Actions)

- **CLI Commands** — Execute shell commands
- **GUI Operations** — Mouse/keyboard via automation
- **API Calls** — External service integration
- **World Model Updates** — State persistence

## Current Status

MVP phase: single pipeline, Mac mini as production environment.

## Tech Stack

- Language: Python 3
- Automation: `screencapture`, `cliclick`, AppleScript
- AI: Anthropic Claude via API
- Storage: JSON world model
- Deployment: VPS (dev), Mac mini (prod)
