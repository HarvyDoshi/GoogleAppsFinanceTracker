import math
import random
import time
import wandb

# ── Simulated Learning Dynamics ────────────────────────────────────────────────
# We simulate a 3-phase learning process typical of RLHF/GRPO:
# 1. Exploration (0-100): Random actions, bad format, reward hacking attempts.
# 2. Scaffolding (100-250): Learns JSON format, basic planning, and retrieval.
# 3. Mastery (250-400): High correctness, verifies data, optimal steps.

def simulate_training_run(total_steps=400):
    wandb.init(
        project="trace-grpo",
        name="full-reward-simulation",
        config={
            "mode": "simulation",
            "model": "Qwen2.5-3B-Instruct-GRPO",
            "total_steps": total_steps,
            "note": "Comprehensive simulated GRPO training run showing multi-component rewards."
        },
    )

    for step in range(total_steps):
        # Progress 0.0 to 1.0
        progress = step / total_steps 
        
        # Add some noise to make graphs look realistic
        noise = lambda scale: random.uniform(-scale, scale)
        
        # ── 1. Format Compliance (Learns very quickly)
        # Starts at 0.2, reaches 1.0 by 20% progress
        format_compliance = min(1.0, 0.2 + (progress * 4) + noise(0.1))
        format_compliance = max(0.0, format_compliance)
        
        # ── 2. Plan Quality (Learns steadily)
        plan_quality = min(0.3, progress * 0.4 + noise(0.05))
        plan_quality = max(0.0, plan_quality)
        
        # ── 3. Retrieval Coverage (Learns after formatting)
        retrieval_coverage = 0.0
        if progress > 0.15:
            retrieval_coverage = min(0.2, (progress - 0.15) * 0.3 + noise(0.05))
        retrieval_coverage = max(0.0, retrieval_coverage)

        # ── 4. Process Reward (PLAN -> RETRIEVE -> VERIFY -> ANSWER)
        process_reward = 0.0
        if progress > 0.3:
            process_reward = min(0.1, (progress - 0.3) * 0.2 + noise(0.02))
        process_reward = max(0.0, process_reward)

        # ── 5. Answer Correctness (Learns late, depends on retrieval)
        answer_correctness = 0.0
        if progress > 0.4:
            # Sudden jumps in understanding (typical in math/logic tasks)
            base_correctness = 0.8 if random.random() < (progress - 0.2) else 0.2
            answer_correctness = min(1.0, base_correctness + noise(0.2))
        answer_correctness = max(0.0, answer_correctness)

        # ── 6. Step Efficiency (Learns to do it faster over time)
        step_efficiency = min(0.15, progress * 0.2 + noise(0.03))
        step_efficiency = max(0.0, step_efficiency)

        # ── 7. Verification Bonus (Only late in training)
        verification_bonus = 0.15 if progress > 0.6 and random.random() > 0.3 else 0.0

        # ── 8. Anti-Hack Penalty (Model tries to cheat early on)
        anti_hack_penalty = 0.0
        # High chance of hacking early, drops to near zero later
        hack_prob = 0.3 * (1.0 - progress)**2
        if random.random() < hack_prob:
            anti_hack_penalty = -0.5 + noise(0.1)

        # Calculate Total Reward (sum of components + penalties)
        total_reward = (
            format_compliance * 0.3 + # Format is a multiplier or base in reality, here we weight it
            plan_quality + 
            retrieval_coverage + 
            process_reward + 
            answer_correctness * 1.5 + # High weight for actual correctness
            step_efficiency + 
            verification_bonus + 
            anti_hack_penalty
        )
        
        # Clip total reward to typical bounds [-1.0, 2.5]
        total_reward = max(-1.0, min(2.5, total_reward))

        # Additional metrics
        loss = max(0.1, 2.5 * math.exp(-progress * 4) + noise(0.2))
        kl_divergence = 0.01 + progress * 0.04 + noise(0.005) # KL drifts up as model deviates from base

        # Log everything to WandB
        wandb.log({
            "reward/total": total_reward,
            "reward_components/format_compliance": format_compliance,
            "reward_components/plan_quality": plan_quality,
            "reward_components/retrieval_coverage": retrieval_coverage,
            "reward_components/process_reward": process_reward,
            "reward_components/answer_correctness": answer_correctness,
            "reward_components/step_efficiency": step_efficiency,
            "reward_components/verification_bonus": verification_bonus,
            "penalties/anti_hack": anti_hack_penalty,
            "training/loss": loss,
            "training/kl_divergence": kl_divergence,
            "epoch": progress * 3.0 # Simulate 3 epochs
        }, step=step)

        if step % 20 == 0:
            print(f"Step {step:03d} | Total Reward: {total_reward:+.3f} | Correctness: {answer_correctness:.2f} | Hack Pen: {anti_hack_penalty:.2f}")

    wandb.finish()
    print("Simulation complete! Check your W&B dashboard for the new graphs.")

if __name__ == "__main__":
    simulate_training_run()
