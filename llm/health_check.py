"""
Manual diagnostic: checks which configured providers are currently reachable.
Run this yourself when you want to check provider status - it is NOT called
automatically by the graph on every request, since that would waste quota.

Usage:
    python -m llm.health_check
"""
from llm.manager import check_provider_health

if __name__ == "__main__":
    print("Checking provider health...\n")
    results = check_provider_health()
    for provider, status in results.items():
        print(f"  {provider}: {status}")
