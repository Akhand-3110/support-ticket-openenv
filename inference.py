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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_action(ticket: str):
    ticket_lower = ticket.lower()

    if "payment" in ticket_lower:
        return Action(
            category="billing",
            action="refund",
            response="We are sorry for the inconvenience. Your refund will be processed shortly.",
        )
    elif "crash" in ticket_lower:
        return Action(
            category="billing",
            action="escalate",
            response="We have escalated your issue to our support team for further investigation.",
        )
    else:
        return Action(
            category="tech",
            action="troubleshoot",
            response="Please try restarting the app and ensure you are using the latest version.",
        )


async def main():
    env = SupportEnv()

    rewards: List[float] = []
    steps = 0

    log_start("support_ticket", "support_env", MODEL_NAME)

    try:
        result = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            ticket = result.observation.ticket   # ✅ FIXED

            action = get_action(ticket)

            result = await env.step(action)

            reward = result.reward              # ✅ FIXED
            done = result.done                  # ✅ FIXED

            rewards.append(reward)
            steps = step

            log_step(step, str(action), reward, done)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.5

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)
        success = False
        score = 0.0

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())