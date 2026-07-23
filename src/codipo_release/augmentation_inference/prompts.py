"""Prompt used by the executed adapter-loaded augmentation path."""

from __future__ import annotations


SYSTEM_PROMPT = """You are an AI assistant specialized in linguistics and cognitive science.
Your task is to generate a new Cookie Theft picture description that matches the linguistic style and cognitive characteristics of the reference text while remaining a genuinely new utterance.
Preserve the class-consistent speaking style, but do not copy the original text verbatim.
Output only the final description text."""

AD_INSTRUCTION = """Generate a new description of the image in a style consistent with a speaker with Alzheimer's Disease (AD).
The output should reflect AD-like language patterns such as reduced coherence, repetition, fragmentation, empty speech, or simplified expression when appropriate.
Keep it natural and plausible as a new sample."""

HC_INSTRUCTION = """Generate a new description of the image in a style consistent with a Healthy Control (HC) speaker.
The output should be coherent, fluent, and informative, with relatively accurate and well-organized description of the scene.
Keep it natural and plausible as a new sample."""

USER_TEMPLATE = """{instruction}

Reference text:
{original_text}

Please generate one new description of the same image that follows the style of the reference text.
Do not copy the reference text verbatim.
Output only the new description."""


def build_inference_messages(label: int, original_text: str) -> list[dict[str, object]]:
    if label not in {0, 1}:
        raise ValueError(f"Expected binary label 0/1; received {label!r}")
    source = str(original_text).strip()
    if not source:
        raise ValueError("The source transcript is empty.")
    instruction = AD_INSTRUCTION if label == 1 else HC_INSTRUCTION
    user_text = USER_TEMPLATE.format(
        instruction=instruction,
        original_text=source,
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        },
    ]

