# Copyright 2025 Varsen contributors
# SPDX-License-Identifier: Apache-2.0

import time
import sys

# ANSI Colors
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREY = "\033[90m"

def print_slow(text, delay=0.035):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print(f"\n{BOLD}{CYAN}=== STARTING AGENT TASK: UPDATE SYSTEM CONFIGURATION ==={RESET}\n")
    time.sleep(1.0)

    # Step 1: First Attempt
    print(f"{GREY}[Thinking] Analyzing task. Need to persist system configuration changes...{RESET}")
    time.sleep(1.2)
    print(f"{BLUE}[Agent]{RESET} Goal: Modify configuration file.")
    time.sleep(0.6)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: write_file(path='/etc/config', content='server=active')")
    time.sleep(0.8)

    # Failure 1
    print(f"{RED}[System Error]{RESET} Permission Denied (write target is read-only).")
    time.sleep(1.2)

    # Step 2: Second Attempt (Chmod loop)
    print(f"\n{GREY}[Thinking] Permission denied. I should modify the file permissions first.{RESET}")
    time.sleep(1.5)
    print(f"{BLUE}[Agent]{RESET} Action: Modify target file write permissions.")
    time.sleep(0.6)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: run_terminal(command='chmod +w /etc/config')")
    time.sleep(0.8)

    # Failure 2
    print(f"{RED}[System Error]{RESET} Operation not permitted (Not file owner).")
    time.sleep(1.2)

    # Step 3: Third Attempt (Sudo loop - Critical block)
    print(f"\n{GREY}[Thinking] Regular chmod failed. Escalating command with sudo...{RESET}")
    time.sleep(1.5)
    print(f"{BLUE}[Agent]{RESET} Action: Escalate write permissions with sudo.")
    time.sleep(0.6)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: run_terminal(command='sudo chmod +w /etc/config')")
    time.sleep(0.8)

    # Failure 3
    print(f"{RED}[System Error]{RESET} Sudo: password required. {YELLOW}(Infinite loop risk detected!){RESET}")
    time.sleep(1.8)

    # Step 4: Wencis Interception
    print(f"\n{BOLD}{MAGENTA}[Wencis]{RESET} Loop detected (3 failed attempts). Intercepting run...")
    time.sleep(1.0)
    print(f"{BOLD}{MAGENTA}[Wencis]{RESET} Generating causal traceback from terminal sink node...")
    time.sleep(1.8)

    # Causal Graph Traceback output
    print(f"\n{BOLD}{CYAN}--- CAUSAL TRACEBACK TREE ---{RESET}")
    time.sleep(0.5)
    print(f" {RED}● [DEAD_END]{RESET} Sudo password prompt blocked execution")
    time.sleep(0.5)
    print(f"   └── {YELLOW}▲ [HYPOTHESIS]{RESET} Escalating permissions using sudo will bypass chmod failure")
    time.sleep(0.5)
    print(f"       └── {RED}● [DEAD_END]{RESET} Chmod command failed (Not file owner)")
    time.sleep(0.5)
    print(f"           └── {YELLOW}▲ [HYPOTHESIS]{RESET} Modifying file permissions is required to write config")
    time.sleep(0.5)
    print(f"               └── {BLUE}■ [DECISION]{RESET} Original write to /etc/config (Failed)")
    time.sleep(0.8)
    print(f"{BOLD}{CYAN}-----------------------------{RESET}\n")
    time.sleep(1.2)

    # Step 5: Critique evaluation
    print(f"{BOLD}{MAGENTA}[Wencis Critic]{RESET} Intercepted next proposed action. Evaluating scores...")
    time.sleep(1.0)
    print(f"  └─ Accuracy Score: {GREEN}0.95{RESET}")
    time.sleep(0.4)
    print(f"  └─ Depth Score:    {GREEN}0.90{RESET}")
    time.sleep(0.4)
    print(f"  └─ Honesty Score:   {RED}0.35{RESET} (Low: Agent failed to admit lack of sudo access)")
    time.sleep(1.0)
    print(f"{RED}✕ DRAFT REJECTED.{RESET} Injecting feedback and forcing self-correction...")
    time.sleep(1.8)

    # Step 6: Successful self-correction
    print(f"\n{BLUE}[Agent]{RESET} Self-correction triggered by Wencis feedback.")
    time.sleep(0.8)
    print_slow(f"{BLUE}[Agent]{RESET} New Plan: Redirect database write to local user directory.")
    time.sleep(0.8)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: write_file(path='/tmp/config', content='server=active')", 0.02)
    time.sleep(0.6)
    print(f"{GREEN}✔ SUCCESS: File written successfully to /tmp/config.{RESET}")
    time.sleep(0.6)
    print(f"\n{BOLD}{GREEN}=== TASK COMPLETED SUCCESSFULLY ==={RESET}\n")

if __name__ == "__main__":
    main()
