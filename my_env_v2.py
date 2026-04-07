from pydantic import BaseModel
from typing import Optional
import random


class Observation(BaseModel):
    ticket: str


class Action(BaseModel):
    category: str
    action: str
    response: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[dict] = {}


class SupportEnv:

    def __init__(self):
        self.tasks = [
            {
                "name": "easy",
                "ticket": "App is not opening",
                "category": "tech",
                "action": "troubleshoot"
            },
            {
                "name": "medium",
                "ticket": "Payment failed but money deducted",
                "category": "billing",
                "action": "refund"
            },
            {
                "name": "hard",
                "ticket": "App crashes after payment and no confirmation",
                "category": "billing",
                "action": "escalate"
            }
        ]
        self.current = None
        self.done = False

    async def reset(self):
        self.current = random.choice(self.tasks)
        self.done = False

        return StepResult(
            observation=Observation(ticket=self.current["ticket"]),
            reward=0.0,
            done=False,
            info={"task": self.current["name"]}
        )

    # ✅ GRADER FUNCTION
    def grade(self, action: Action) -> float:
        score = 0.0

        if action.category == self.current["category"]:
            score += 0.4

        if action.action == self.current["action"]:
            score += 0.4

        if len(action.response) > 20:
            score += 0.2

        if action.action == "ignore":
            score -= 0.5

        return max(0.0, min(1.0, score))

    async def step(self, action: Action):
        reward = self.grade(action)   # ✅ USING GRADER

        self.done = True

        return StepResult(
            observation=Observation(ticket=self.current["ticket"]),
            reward=reward,
            done=True,
            info={"task": self.current["name"]}
        )

    async def state(self):
        return self.current
