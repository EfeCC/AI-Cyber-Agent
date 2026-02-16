import sys
import os
import yaml
from colorama import Fore

# Force the correct path to the front
module_path = os.path.abspath(os.path.join("system", "PentestAiAgent"))
sys.path.insert(0, module_path)

from main import PentestAgent

def test_api_key():
    print("Checking API Key detection...")
    try:
        # We need to be in the folder where config.yaml is, or hope CWD works
        agent = PentestAgent("http://localhost:8080")
        if agent.cve_checker.api_key:
            print(f"SUCCESS: API Key found: {agent.cve_checker.api_key[:5]}...")
        else:
            print("FAILURE: API Key NOT found in agent.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_api_key()
