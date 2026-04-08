from fastapi import FastAPI
from env import EmailTriageEnv
from models import Action

app = FastAPI()

env = EmailTriageEnv()


@app.get("/")
def home():
    return {"status": "ok"}


@app.get("/test")
def test():
    return {"status": "working"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {
    "tasks": [
        {
            "name": "refund",
            "description": "Handle refund request email",
            "difficulty": "easy"
        },
        {
            "name": "complaint",
            "description": "Resolve customer complaint",
            "difficulty": "medium"
        },
        {
            "name": "angry",
            "description": "Handle angry customer professionally",
            "difficulty": "hard"
        }
    ],
    "action_schema": {
        "category": "string (refund | complaint | angry)",
        "reply": "string"
    }
}


@app.get("/baseline")
def baseline():
    scores = []

    test_actions = [
        Action(category="refund", reply="You can request a refund through our policy process."),
        Action(category="complaint", reply="Sorry for the issue, we will resolve it quickly."),
        Action(category="angry", reply="We apologize and understand your frustration, we will help.")
    ]

    for action in test_actions:
        env.reset()
        _, reward, _, _ = env.step(action)
        scores.append(reward.score)

    return {
        "scores": scores,
        "average_score": sum(scores) / len(scores)
    }


@app.get("/grader")
def grader():
    return {"message": "Grader runs inside step() reward system"}