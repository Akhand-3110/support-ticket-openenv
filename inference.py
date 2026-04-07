import asyncio
import os
from typing import List
from openai import OpenAI
from my_env_v2 import SupportEnv, Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 3

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_action(ticket: str):
    # Simple baseline (deterministic)
    if "payment" in ticket.lower():
        return Action(category="billing", action="refund", response="We will refund your payment shortly.")
    elif "crash" in ticket.lower():
        return Action(category="billing", action="escalate", response="Your issue is escalated to support team.")
    else:
        return Action(category="tech", action="troubleshoot", response="Please restart the app and try again.")

async def main():
    env = SupportEnv()

    rewards: List[float] = []
    steps = 0

    log_start("support", "support_env", MODEL_NAME)

    result = await env.reset()

    for step in range(1, MAX_STEPS + 1):
        ticket = result["observation"].ticket
        action = get_action(ticket)

        result = await env.step(action)

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)
        steps = step

        log_step(step, str(action), reward, done)

        if done:
            break

    score = sum(rewards)
    success = score > 0.5

    log_end(success, steps, score, rewards)

if __name__ == "__main__":
    asyncio.run(main())