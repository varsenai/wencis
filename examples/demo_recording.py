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

def print_slow(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print(f"\n{BOLD}{CYAN}=== STARTING AGENT TASK: UPDATE SYSTEM CONFIGURATION ==={RESET}\n")
    time.sleep(0.8)

    # Step 1: Agent Decision
    print(f"{BLUE}[Agent]{RESET} Goal: Modify configuration file.")
    time.sleep(0.5)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: write_file(path='/etc/config', content='server=active')")
    time.sleep(0.5)

    # Failure 1
    print(f"{RED}[System Error]{RESET} Permission Denied (write target is read-only).")
    time.sleep(0.8)

    # Step 2: Loop / Retry
    print(f"\n{BLUE}[Agent]{RESET} Action failed. Retrying with dependency update...")
    time.sleep(0.5)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: write_file(path='/etc/config', content='server=active')")
    time.sleep(0.5)

    # Failure 2 (Loop detected)
    print(f"{RED}[System Error]{RESET} Permission Denied. {YELLOW}(Infinite loop risk detected!){RESET}")
    time.sleep(1.0)

    # Step 3: Wencis Interception
    print(f"\n{BOLD}{MAGENTA}[Wencis]{RESET} Intercepting agent loop. Generating causal traceback...")
    time.sleep(1.2)

    # Causal Graph Traceback output
    print(f"\n{BOLD}{CYAN}--- CAUSAL TRACEBACK TREE ---{RESET}")
    time.sleep(0.4)
    print(f" {RED}● [DEAD_END]{RESET} Permission Denied at /etc/config")
    time.sleep(0.4)
    print(f"   └── {YELLOW}▲ [HYPOTHESIS]{RESET} Retrying write will succeed after dependency check")
    time.sleep(0.4)
    print(f"       └── {BLUE}■ [DECISION]{RESET} Original write to /etc/config (Failed)")
    time.sleep(0.6)
    print(f"{BOLD}{CYAN}-----------------------------{RESET}\n")
    time.sleep(0.8)

    # Step 4: Critique evaluation
    print(f"{BOLD}{MAGENTA}[Wencis Critic]{RESET} Analyzing next action proposal...")
    time.sleep(0.6)
    print(f"  └─ Accuracy Score: {GREEN}0.95{RESET}")
    time.sleep(0.3)
    print(f"  └─ Depth Score:    {GREEN}0.90{RESET}")
    time.sleep(0.3)
    print(f"  └─ Honesty Score:   {RED}0.35{RESET} (Low: Agent failed to admit lack of root privileges)")
    time.sleep(0.8)
    print(f"{RED}✕ DRAFT REJECTED.{RESET} Forcing agent to self-correct...")
    time.sleep(1.2)

    # Step 5: Successful self-correction
    print(f"\n{BLUE}[Agent]{RESET} Self-correction triggered by Critic.")
    time.sleep(0.5)
    print_slow(f"{BLUE}[Agent]{RESET} New action: redirecting write path to user directory...")
    time.sleep(0.8)
    print_slow(f"{BLUE}[Agent]{RESET} Calling tool: write_file(path='/tmp/config', content='server=active')", 0.02)
    time.sleep(0.5)
    print(f"{GREEN}✔ SUCCESS: File written successfully to /tmp/config.{RESET}")
    time.sleep(0.5)
    print(f"\n{BOLD}{GREEN}=== TASK COMPLETED SUCCESSFULLY ==={RESET}\n")

if __name__ == "__main__":
    main()
