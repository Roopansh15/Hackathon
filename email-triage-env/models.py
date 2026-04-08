from pydantic import BaseModel

class Observation(BaseModel):
    email_text: str

class Action(BaseModel):
    category: str
    reply: str

class Reward(BaseModel):
    score: float