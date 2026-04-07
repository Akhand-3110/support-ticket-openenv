from fastapi import FastAPI
from my_env_v2 import SupportEnv, Action
import uvicorn

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


# ✅ REQUIRED FOR OPENENV VALIDATION
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED ENTRY POINT
if __name__ == "__main__":
    main()