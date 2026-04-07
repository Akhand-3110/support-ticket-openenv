import asyncio
import os
from typing import List
from openai import OpenAI
from my_env_v2 import SupportEnv, Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 3

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


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


def get_action_from_llm(ticket: str) -> Action:
    prompt = f"""
You are a customer support agent.

Ticket:
{ticket}

Respond in JSON format:
{{
  "category": "...",
  "action": "...",
  "response": "..."
}}

Rules:
- category: billing / tech / general
- action: refund / troubleshoot / escalate / ignore
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100,
    )

    text = response.choices[0].message.content.strip()

    # simple safe fallback
    try:
        import json
        data = json.loads(text)
        return Action(**data)
    except:
        return Action(
            category="general",
            action="escalate",
            response="We are looking into your issue."
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

            ticket = result.observation.ticket

            action = get_action_from_llm(ticket)   # ✅ LLM CALL

            result = await env.step(action)

            reward = result.reward
            done = result.done

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
