import os
import sys
import subprocess
import requests
from typing import List, Optional
from openai import OpenAI


def bootstrap():
    deps = ["openai", "requests"]
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--quiet"])

bootstrap()


BASE_URL = "https://roopanshsaxena-email-triage-env.hf.space"

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 2
SUCCESS_SCORE_THRESHOLD = 0.3

def get_client():
    api_key = os.getenv("HF_TOKEN")

    if not api_key:
        print("[DEBUG] No HF_TOKEN found, using dummy mode", file=sys.stderr)
        return None

    return OpenAI(
        base_url=os.getenv("API_BASE_URL"),
        api_key=api_key
    )


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env=email-triage model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(email_text):
    client = get_client()

    if client is None:
        text = email_text.lower()

        if "refund" in text:
            return "refund", "We are sorry. You can request a refund through our refund policy and process."
        elif "bad" in text or "quality" in text:
            return "complaint", "We are sorry for the issue. We will resolve your complaint as soon as possible."
        elif "angry" in text:
            return "angry", "We sincerely apologize. We understand your frustration and will help you immediately."

        return "complaint", "We are sorry. We will resolve your issue."

    
    try:
        res = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[
                {"role": "system", "content": "Classify email into refund, complaint, or angry and reply."},
                {"role": "user", "content": email_text}
            ],
            temperature=0.3,
            max_tokens=50
        )

        text = res.choices[0].message.content.lower()

        if "refund" in text:
            return "refund", text
        elif "complaint" in text:
            return "complaint", text
        else:
            return "angry", text

    except Exception as e:
        print(f"[DEBUG] {e}", file=sys.stderr)
        return "complaint", "We are sorry. We will resolve your issue."


def run_task(task_name):
    rewards = []
    steps = 0
    score = 0.0
    success = False

    log_start(task_name, "email-triage", MODEL_NAME)

    try:
        res = requests.post(f"{BASE_URL}/reset")
        obs = res.json()
        email = obs["email_text"]

        for step in range(1, MAX_STEPS + 1):
            category, reply = get_action(email)

            action = {
                "category": category,
                "reply": reply
            }

            res = requests.post(f"{BASE_URL}/step", json=action)
            data = res.json()

            reward = data["reward"]["score"]

            if reward <= 0.0:
                reward = 0.01
            elif reward >= 1.0:
                reward = 0.99
            done = data["done"]

            rewards.append(reward)
            steps = step

            log_step(step, category, reward, done, None)

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0

        # clamp score strictly between (0,1)
        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps, score, rewards)


def main():
    tasks = ["refund-task", "complaint-task", "angry-task"]

    for task in tasks:
        run_task(task)

if __name__ == "__main__":
    main()