# 🧠 Silex Cognitive Engine

Silex is a proprietary, closed-source cognitive engine developed by OpenYF. It powers the public **VYN** client through a high-performance, background asynchronous JSON-RPC daemon.

## 🏗️ The Interface-Separation Architecture

Silex operates entirely headless and isolated from any front-end UI. 
To protect the intellectual property of the core engine while allowing open-source collaboration on the frontend, we use the **Interface-Separation Pattern**:

1. **The Daemon**: This repository compiles down to a single standalone executable (`silex.exe`).
2. **The RPC Loop**: When booted, `silex.exe` runs a pure Python `http.server` background daemon on `localhost:8080`.
3. **The Client**: The open-source VYN CLI sends user messages to the daemon via JSON HTTP `POST /cognitive`.
4. **The Response**: Silex computes the response using its private Causal Knowledge Graph, Memory Store, and Meta-Reasoning systems, and returns the structured data to VYN to be rendered.

## 🚀 Getting Started

### Local Development

Because Silex is completely decoupled, you can develop and test it by passing JSON payloads directly to the daemon.

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Boot the Daemon**
   ```bash
   python src/silex/server/daemon.py
   ```

3. **Ping the Engine (cURL)**
   ```bash
   curl -X GET http://127.0.0.1:8080/health
   ```

4. **Send a Query**
   ```bash
   curl -X POST http://127.0.0.1:8080/cognitive \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello Silex, remember my name is Operator."}'
   ```

## 📦 Build & Release Pipeline

This repository contains a GitHub Actions workflow (`.github/workflows/build-silex.yml`) that automatically triggers when you push a version tag (e.g. `v2.0.0`). 

The pipeline uses **PyInstaller** to compile `daemon.py` into native binaries for Windows (`silex.exe`), macOS, and Linux. It then automatically uploads those closed-source binaries to the public VYN repository's Releases page so users can download them via the `curl | bash` installation script.
