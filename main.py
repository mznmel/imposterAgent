from dotenv import load_dotenv
import dspy
import os
import random

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

lm = dspy.LM(
    "openrouter/google/gemini-3-flash-preview",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
dspy.configure(lm=lm)

SECRET_WORDS = [
    "desert","ocean","moon","sun","rain","snow","mountain","river","city","village",
    "school","hospital","airport","market","book","pen","clock","mirror","window","door",
    "car","train","airplane","ship","garden","flower","tree","apple","coffee","tea",
    "bread","salt","sugar","honey","fire","air","sand","gold","silver","diamond",
    "computer","phone","camera","key","map","sound","light","shadow","dream","movie",
    "song","music","painting","picture","language","heart","mind","time","holiday","kingdom",
    "science","history","intelligence","programming","algorithm","robot","network","library","museum","stadium",
    "ball","swimming","running","bridge","road","lake","forest","island","palm","date",
    "milk","horse","camel","tiger","bird","house","palace","tower","hammer","knife",
    "kitchen","office","bag","chair","bed","star","cloud","wind","storm","glass",
]

num_rounds = 3
num_agents = random.randint(5,8)
imposter_agent_id = random.randint(0, num_agents - 1)
secret_word = random.choice(SECRET_WORDS)
agents = [
    {"d": i, "descriptions": [], "role": "IMPOSTER" if i == imposter_agent_id else "NON_IMPOSTER"}
    for i in range(num_agents)
]

# DSPy Signatures

class Turn(dspy.Signature):
    """You are an Agent in a multi-agent social deduction game.
    In each round, every player must describe a secret word using three words or fewer.
    No one is allowed to mention the word itself.
    No one may repeat any previous description used in the game.
    After all rounds are completed, all players will vote to determine who the Imposter is.

    If role is NON_IMPOSTER:
    The secret word will be provided to you.
    Your goal is to describe the word subtly without making it too obvious and detect the Imposter.

    If role is IMPOSTER:
    You are not given the secret word.
    Your goal is to blend in, give plausible descriptions, and infer the word without being detected.

    Response rules:
    - Maximum three sentences.
    - Do not mention the rules.
    - Do not mention your role.
    - Do not mention the secret word."""

    num_rounds: int = dspy.InputField(desc="Number of rounds")
    current_round: int = dspy.InputField(desc="Current round")
    role: str = dspy.InputField(desc="Your role")
    agent_id: int = dspy.InputField(desc="Your agent ID")
    secret_word: str = dspy.InputField(desc="The secret word")
    all_descriptions: str = dspy.InputField(desc="All descriptions given so far by all players")
    description: str = dspy.OutputField(desc="Your description in 3 words or fewer")

class VoteImposter(dspy.Signature):
    """You are in a multi-agent social deduction game. 
    You are voting to determine who the Imposter is.
    Players described a secret word using three words or fewer.
    No one was allowed to mention the word itself.
    No one was allowed to repeat any previous description used in the game.

    If role is NON_IMPOSTER:
    The secret word was provided to the player.
    The player's goal was to describe the word subtly without making it too obvious and detect the Imposter.

    If role is IMPOSTER:
    The player was not given the secret word.
    The player's goal is to blend in, give plausible descriptions, and infer the word without being detected.

    Analyze all descriptions carefully and vote for the agent you believe is the Imposter.
    If you are the imposter, vote for another agent.
    """
    num_rounds: int = dspy.InputField(desc="Number of rounds")
    num_agents: int = dspy.InputField(desc="Number of agents")
    role: str = dspy.InputField(desc="Your role")
    agent_id: int = dspy.InputField(desc="Your agent ID")
    secret_word: str = dspy.InputField(desc="Secret word")
    all_descriptions: str = dspy.InputField(desc="Complete log of all descriptions from the 3 rounds")
    vote: int = dspy.OutputField(desc="Single integer: the agent ID you suspect is the Imposter")


print("=" * 60)
print(f"Number of agents: {num_agents}")
print(f"Imposter agent ID: {imposter_agent_id}")
print(f"Secret word: {secret_word}")
print("=" * 60)

all_descriptions = []
turn_predictor = dspy.ChainOfThought(Turn)
vote_predictor = dspy.Predict(VoteImposter)

for round_idx in range(num_rounds):
    print(f"Round {round_idx + 1}")
    for agent in agents:
        response = turn_predictor(num_rounds=num_rounds, current_round=round_idx, role=agent["role"], agent_id=agent["d"], 
        secret_word=secret_word if agent["role"] == "NON_IMPOSTER" else "", all_descriptions="\n".join(all_descriptions))
        description = response.description.strip()
        agent["descriptions"].append(description)
        all_descriptions.append(f"Round# {round_idx + 1}, Agent# {agent['d']}: {description}")
        print(f"Agent {agent['d']}: {description}")

print("=" * 60)
print("Voting")
print("=" * 60)

votes = []
for agent in agents:
    response = vote_predictor(num_rounds=num_rounds, num_agents=num_agents, role=agent["role"], agent_id=agent["d"], 
    secret_word=secret_word if agent["role"] == "NON_IMPOSTER" else "", all_descriptions="\n".join(all_descriptions))
    vote = response.vote
    votes.append(vote)
    print(f"Agent {agent['d']}: {vote}")

# majority vote
majority_vote = max(set(votes), key=votes.count)
print("=" * 60)
print(f"Votes: {votes}")
print(f"Majority vote: {majority_vote}")
print(f"Actual imposter agent ID: {imposter_agent_id}")
print("=" * 60)
