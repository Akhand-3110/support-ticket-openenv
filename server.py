from fastapi import FastAPI
from my_env_v2 import SupportEnv, Action

app = FastAPI()
env = SupportEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
async def reset():
    result = await env.reset()
    return result.dict()

@app.post("/step")
async def step(action: dict):
    result = await env.step(Action(**action))
    return result.dict()