from fastapi import FastAPI
from my_env_v2 import SupportEnv, Action

app = FastAPI()
env = SupportEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
async def reset():
    return await env.reset()

@app.post("/step")
async def step(action: dict):
    return await env.step(Action(**action))