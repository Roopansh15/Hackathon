from env import EmailTriageEnv
from models import Action

env = EmailTriageEnv()

obs = env.reset()
print("Email:", obs.email_text)

action = Action(
    category="refund",
    reply="You can request a refund. We are happy to help."
)

obs, reward, done, info = env.step(action)

print("Reward:", reward.score)