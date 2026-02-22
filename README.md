# 🕵️ imposterAgent

A simple example of how to use [DSPy](https://dspy.ai) to build a multi-agent social deduction game.

## 🎮 How It Works

A group of 5–8 AI agents play a word-guessing game, where one agent is assigned the **Imposter** role.

1. 🎭 One agent is randomly assigned the **Imposter** role.
2. 🔑 All other agents receive the secret word.
3. 💬 Over 3 rounds, each agent describes the word in **3 words or fewer** without saying the word itself.
4. 🗳️ After all rounds, every agent votes on who they think the Imposter is.
5. ⚖️ The majority vote determines the outcome.

The Imposter must blend in by inferring the word from others' descriptions, while non-Imposters must describe subtly enough to avoid giving it away.

## 🧩 DSPy Concepts Used

- 📝 **Signatures** — `Turn` and `VoteImposter` define structured input/output contracts for LLM calls.
- 🤖 **Predictors** — `dspy.ChainOfThought` (for descriptions) and `dspy.Predict` (for voting) handle the LLM interactions.
- ⚙️ **LM configuration** — A single `dspy.LM` instance configured with OpenRouter.

## 🛠️ Setup
Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

## 🚀 Usage

```bash
uv sync
uv run main.py
```
