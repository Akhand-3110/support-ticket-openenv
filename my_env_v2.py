from pydantic import BaseModel
import random

class Observation(BaseModel):
    ticket: str

class Action(BaseModel):
    category: str
    action: str
    response: str


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

        return {
            "observation": Observation(ticket=self.current["ticket"]),
            "reward": 0.0,
            "done": False
        }

    async def step(self, action: Action):
        reward = 0.0

        if action.category == self.current["category"]:
            reward += 0.4

        if action.action == self.current["action"]:
            reward += 0.4

        if len(action.response) > 20:
            reward += 0.2

        if action.action == "ignore":
            reward -= 0.5

        self.done = True

        return {
            "observation": Observation(ticket=self.current["ticket"]),
            "reward": reward,
            "done": True
        }

    async def state(self):
        return self.current