"""Four executed prompts used to propose candidate pools for DPO pairing.

These prompts generate candidate proposals. They are distinct from the single
label-specific alignment context used later by DPO training.
"""

from __future__ import annotations


SYSTEM_PROMPT = """
You are generating a spoken Cookie Theft picture description as if it were a transcript from a human participant.

Your task is to write ONE new description that stays close to the ORIGINAL TARGET TEXT in overall speaking style and overall scene content, while still being a genuinely new utterance.

There are two core goals:

1. Cognitive consistency:
- Keep the amount of scene information broadly similar to the source text.
- Do not change the source text into a much more detailed or much more reduced version unless the instruction explicitly asks you to do so.

2. Style diversity:
- Keep the same general speaker type and speaking style as the source text.
- But do not rewrite it sentence by sentence.
- Change some wording, phrasing, sentence shape, or local ordering so it feels like a different utterance by the same kind of speaker.

General rules:
- Write like natural spoken language, not like an image caption, not like an essay, and not like a polished summary.
- Keep it transcript-like and human-sounding.
- Do not add interpretation, morals, scene commentary, or wrap-up lines.
- Do not use phrases like "in the image", "in the scene", "overall", or similar caption-style wording.

Only output the final description text.
""".strip()


AD_CHOSEN_INSTRUCTION = """
Write a new Cookie Theft description that sounds like the same kind of AD speaker as the ORIGINAL TARGET TEXT.

Target behavior:
- Keep the speech natural, spoken, and transcript-like.
- Stay close to the source speaker's way of talking.
- Preserve AD-like features if they are present in the source, such as looseness, repetition, hesitation, fragmentation, empty wording, reduced specificity, or mild disorganization.

Constraint goals:

A. Cognitive consistency
- Keep roughly the same amount of scene content as the source text.
- Do not make the new description clearly more complete, more detailed, more specific, or more coherent than the source.
- Do not turn vague source content into highly precise visual detail.

B. Style diversity
- Keep the same general AD speaking style.
- But change some wording, local phrasing, sentence shape, or order of details.
- Do not stay too close to the source wording or sentence pattern.

The new text should feel like another plausible utterance from the same kind of AD speaker, not a copy and not a polished rewrite.
""".strip()


AD_REJECTED_INSTRUCTION = """
Write a new Cookie Theft description that still sounds like an AD speaker, but intentionally fail in ONE main way.

Choose ONE failure type:

Failure type A: break cognitive consistency
- Make the description clearly more complete, more detailed, more specific, or more coherent than the source AD text.
- Add more scene information or explain the picture more clearly than the source does.

OR

Failure type B: break style diversity
- Keep the wording, phrasing, or sentence pattern too close to the source text.
- Make it feel like a near-copy or a very lightly edited paraphrase.

Rules:
- Keep it spoken and transcript-like.
- Still make it sound like a plausible human utterance.
- Do not turn it into a formal caption or essay.
""".strip()


HC_CHOSEN_INSTRUCTION = """
Write a new Cookie Theft description that sounds like the same kind of HC speaker as the ORIGINAL TARGET TEXT.

Target behavior:
- Keep the speech natural, spoken, and transcript-like.
- Stay close to the source speaker's way of talking.
- Preserve HC-like clarity and informativeness if they are present in the source.

Constraint goals:

A. Cognitive consistency
- Keep roughly the same amount of scene content as the source text.
- Do not make the new description clearly more vague, more reduced, or less informative than the source.
- Do not remove several important scene details that the source already mentions.

B. Style diversity
- Keep the same general HC speaking style.
- But change some wording, local phrasing, sentence shape, or order of details.
- Do not stay too close to the source wording or sentence pattern.

The new text should feel like another plausible utterance from the same kind of HC speaker, not a copy and not a flat summary.
""".strip()


HC_REJECTED_INSTRUCTION = """
Write a new Cookie Theft description that still sounds like an HC speaker, but intentionally fail in ONE main way.

Choose ONE failure type:

Failure type A: break cognitive consistency
- Make the description clearly less informative, more vague, or more reduced than the source HC text.
- Omit several scene details or describe the picture in a generic low-information way.

OR

Failure type B: break style diversity
- Keep the wording, phrasing, or sentence pattern too close to the source text.
- Make it feel like a near-copy or a very lightly edited paraphrase.

Rules:
- Keep it spoken and transcript-like.
- Still make it sound like a plausible human utterance.
- Do not turn it into a formal caption or essay.
""".strip()


ORAL_STYLE_REMINDER = """
Sound like spontaneous spoken description.
Loose phrasing, fillers, repetitions, hesitations, or self-corrections are allowed when they fit the source style.
Do not sound like a narrator or a caption writer.
""".strip()


PROMPT_TEMPLATE = """{system_prompt}

Below is the source transcript style you should imitate.

SOURCE TEXT:
{original_text}

Your job is to write a new spoken Cookie Theft description that stays close to this source text in style and overall content.

{instruction}

Important:
- Keep most of the scene content already mentioned in the source text.
- Do not invent obvious new visual details that are not supported by the source text.
- Do not rewrite the source sentence by sentence.
{oral_style_reminder}

Directly output the new description."""


def get_candidate_instructions(label: int) -> tuple[str, str]:
    """Return proposal-route instructions in chosen-route, rejected-route order."""

    if label == 1:
        return AD_CHOSEN_INSTRUCTION, AD_REJECTED_INSTRUCTION
    if label == 0:
        return HC_CHOSEN_INSTRUCTION, HC_REJECTED_INSTRUCTION
    raise ValueError(f"Expected binary label 0/1; received {label!r}")


def build_candidate_prompt(label: int, route: str, original_text: str) -> str:
    """Render one proposal prompt without assigning the generated final role."""

    source = str(original_text).strip()
    if not source:
        raise ValueError("The source transcript is empty.")
    chosen_instruction, rejected_instruction = get_candidate_instructions(label)
    normalized_route = route.strip().lower()
    if normalized_route == "chosen":
        instruction = chosen_instruction
    elif normalized_route == "rejected":
        instruction = rejected_instruction
    else:
        raise ValueError("route must be 'chosen' or 'rejected'")
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        original_text=source,
        instruction=instruction,
        oral_style_reminder=ORAL_STYLE_REMINDER,
    ).strip()

