"""Final prompt families for the three ICL augmentation controls."""

from __future__ import annotations


SYSTEM_PROMPTS = {
    "rewrite": (
        "You are given the Cookie Theft picture and one original spoken "
        "description.\nWrite one different spoken description of the same "
        "picture.\n\nThe output should sound like natural spoken language "
        "rather than a formal written caption or summary.\nOutput only the "
        "new description text."
    ),
    "imitation": (
        "You are given the Cookie Theft picture and one reference spoken "
        "description.\nWrite one spoken description of the same picture using "
        "the reference only as an example of how a person might speak.\n\n"
        "The output should sound like natural spoken language rather than a "
        "formal written caption or summary.\nOutput only the new description "
        "text."
    ),
    "direct": (
        "You are given the Cookie Theft picture.\nYour task is to write one "
        "spoken description of the picture as if it were produced by a human "
        "participant.\n\nThe description should sound like natural speech, "
        "not like a polished written caption, not like a formal summary, and "
        "not like an AI explanation.\n\nOutput only the description text."
    ),
}

LABEL_INSTRUCTIONS = {
    ("rewrite", 1): (
        "Generate one spoken description for a speaker with Alzheimer's "
        "Disease (AD).\nIt may naturally contain hesitation, repetition, "
        "fragmentation, vague wording, or reduced coherence."
    ),
    ("rewrite", 0): (
        "Generate one spoken description for a Healthy Control (HC) speaker.\n"
        "It should sound natural, fluent, and reasonably informative as "
        "spoken language."
    ),
    ("imitation", 1): (
        "Generate one spoken description for a speaker with Alzheimer's "
        "Disease (AD).\nIt may naturally contain hesitation, repetition, "
        "fragmentation, vague wording, or reduced coherence."
    ),
    ("imitation", 0): (
        "Generate one spoken description for a Healthy Control (HC) speaker.\n"
        "It should sound natural, fluent, and reasonably informative as "
        "spoken language."
    ),
    ("direct", 1): (
        "Generate one plausible spoken Cookie Theft description for a speaker "
        "with Alzheimer's Disease (AD).\nThe output may naturally include "
        "hesitation, repetition, fragmentation, reduced specificity, or loose "
        "organization when appropriate.\nKeep it plausible as a "
        "human-produced spoken response."
    ),
    ("direct", 0): (
        "Generate one plausible spoken Cookie Theft description for a Healthy "
        "Control (HC) speaker.\nThe output should sound natural, fluent, and "
        "reasonably informative as spoken language.\nKeep it plausible as a "
        "human-produced spoken response."
    ),
}


def build_icl_prompt(
    method: str, label: int, original_text: str | None
) -> tuple[str, str]:
    """Return the system and user text used for one ICL proposal."""
    method = method.strip().lower()
    if method not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported ICL method: {method}")
    if label not in {0, 1}:
        raise ValueError(f"Label must be 0 or 1, found {label}.")
    instruction = LABEL_INSTRUCTIONS[(method, label)]
    if method == "rewrite":
        if original_text is None:
            raise ValueError("Rewrite requires an original transcript.")
        user = (
            f"{instruction}\n\nOriginal text:\n{original_text}\n\n"
            "Write one new spoken description of the same picture as a "
            "rewritten variant of the original text.\nDo not copy the original "
            "text exactly.\nOutput only the new description."
        )
    elif method == "imitation":
        if original_text is None:
            raise ValueError("Imitation requires a reference transcript.")
        user = (
            f"{instruction}\n\nReference text:\n{original_text}\n\n"
            "Write one new spoken description of the same picture that "
            "imitates the speaking style of the reference text.\nIt should be "
            "a genuinely new utterance, not a line-by-line rewrite.\nOutput "
            "only the new description."
        )
    else:
        user = (
            f"{instruction}\n\nWrite one spoken description of the picture.\n"
            "Output only the description."
        )
    return SYSTEM_PROMPTS[method], user

