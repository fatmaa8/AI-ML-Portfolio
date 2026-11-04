
from src.llm import client

MODEL = "openai/gpt-oss-20b"


def run_experiment(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.7,
        max_completion_tokens=300
    )

    print(response.choices[0].message.content)

    print("\nFinish Reason:")
    print(response.choices[0].finish_reason)


def test_max_tokens():
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explain Artificial Intelligence "
                    "in detail with examples."
                )
            }
        ],
        temperature=0.7,
        max_completion_tokens=20
    )

    print("\n--- Limited Token Response ---")
    print(response.choices[0].message.content)

    print("\nFinish Reason:")
    print(response.choices[0].finish_reason)


def test_temperature(temperature, runs=3):
    prompt = "Explain Artificial Intelligence in 3 sentences."

    print(f"\n--- Temperature = {temperature} ---")

    for i in range(1, runs + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=500
        )

        print(f"\nRun {i}:")
        print(response.choices[0].message.content)

        print("\nFinish Reason:")
        print(response.choices[0].finish_reason)


def main():
    prompt = "Explain Artificial Intelligence."

    print("\n--- Experiment 1: Professional Tone ---")
    run_experiment(
        "You are a professional AI instructor. "
        "Explain concepts clearly using formal language.",
        prompt
    )

    print("\n--- Experiment 2: Friendly Tone ---")
    run_experiment(
        "You are a friendly teacher. "
        "Explain concepts in a fun, simple, and conversational way.",
        prompt
    )

    print("\n--- Experiment 3: Academic Tone ---")
    run_experiment(
        "You are an academic researcher. "
        "Explain concepts using precise technical language.",
        prompt
    )

    test_max_tokens()

    test_temperature(temperature=0, runs=3)

    test_temperature(temperature=1, runs=3)


if __name__ == "__main__":
    main()
