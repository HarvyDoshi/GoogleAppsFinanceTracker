"""Quick test script for the Trace Environment API flow using TraceEnvClient."""
import time
import json
from client import TraceEnvClient
from models import TraceAction

BASE = "http://127.0.0.1:8000"

def test_full_flow():
    print("=" * 60)
    print("TEST: Full Trace Environment flow via TraceEnvClient")
    print("=" * 60)

    # Use port 8000 (default)
    with TraceEnvClient(base_url=BASE).sync() as client:
        # 1. Reset
        print("\n[1] Reset episode...")
        result = client.reset()
        obs = result.observation
        print(f"    Episode ID: {obs.episode_id}")
        print(f"    Instruction: {obs.instruction}")
        print(f"    Step: {obs.step}")

        # 2. Wait a bit (simulating user thinking) then PLAN
        print("\n    (Waiting 1 second...)")
        time.sleep(1)

        print("\n[2] Step: PLAN...")
        result = client.step(TraceAction(
            action_type="PLAN",
            content="Step 1: Query Gmail for receipts in 2025. Step 2: Memorize key findings. Step 3: Verify results. Step 4: Submit final answer.",
        ))
        print(f"    Done: {result.done}")
        print(f"    Reward: {result.reward}")
        print(f"    Context: {result.observation.context[:120]}")

        # 3. RETRIEVE from gmail
        print("\n[3] Step: RETRIEVE from gmail...")
        result = client.step(TraceAction(
            action_type="RETRIEVE",
            content="receipt 2025",
            source="gmail",
        ))
        print(f"    Done: {result.done}")
        print(f"    Reward: {result.reward}")
        print(f"    Context: {result.observation.context[:120]}")

        # 4. MEMORIZE
        print("\n[4] Step: MEMORIZE...")
        result = client.step(TraceAction(
            action_type="MEMORIZE",
            content="Found receipts from Gmail for 2025 transactions.",
        ))
        print(f"    Done: {result.done}")
        print(f"    Reward: {result.reward}")

        # 5. VERIFY
        print("\n[5] Step: VERIFY...")
        result = client.step(TraceAction(
            action_type="VERIFY",
            content="Receipts were found in Gmail for 2025.",
        ))
        print(f"    Done: {result.done}")
        print(f"    Reward: {result.reward}")
        print(f"    Verified: {result.observation.context[:120]}")

        # 6. ANSWER
        print("\n[6] Step: ANSWER...")
        result = client.step(TraceAction(
            action_type="ANSWER",
            content="Based on my search of Gmail, I found several receipts from 2025 including order confirmations, invoices, and subscription renewals.",
        ))
        print(f"    Done: {result.done}")
        print(f"    Reward: {result.reward}")
        assert result.done == True

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_flow()
