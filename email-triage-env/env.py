import random
from models import Observation, Action, Reward

class EmailTriageEnv:
    def __init__(self):
        self.tasks = [
            ("I want a refund for my recent purchase", "refund"),
            ("This product quality is very bad", "complaint"),
            ("I am extremely angry about your service", "angry")
        ]
        self.current_email = None
        self.correct_category = None
        self.history = []

    def reset(self):
        self.current_email, self.correct_category = random.choice(self.tasks)
        return Observation(email_text=self.current_email)

    def step(self, action: Action):
        if self.current_email is None:
            return Observation(email_text="No email loaded"), Reward(score=0.0), True, {"error": "Call reset() first"}

        email = self.current_email.lower()
        category = action.category.lower()
        reply = action.reply.lower()

        score = 0.0

        if category == self.correct_category:
            score += 0.5

        keywords = {
            "refund": ["refund", "policy", "process"],
            "complaint": ["sorry", "issue", "resolve"],
            "angry": ["apologize", "understand", "help"]
        }

        if category in keywords:
            for word in keywords[category]:
                if word in reply:
                    score += 0.1

        if len(reply.split()) > 5:
            score += 0.2

        if category not in ["refund", "complaint", "angry"]:
            score -= 0.3

        if len(reply.strip()) == 0:
            score -= 0.2

        score = max(0.0, min(1.0, score))

        self.history.append({
            "email": self.current_email,
            "action": action.model_dump(),
            "score": score
        })

        reward = Reward(score=score)
        done = True

        return Observation(email_text=self.current_email), reward, done, {}

    def state(self):
        return {
            "current_email": self.current_email,
            "correct_category": self.correct_category,
            "history": self.history
        }