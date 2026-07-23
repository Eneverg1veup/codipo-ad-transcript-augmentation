"""Shared DPO conditioning prompt extracted from the executed training path.

These are not the four proposal prompts used to generate pair candidates.
Every chosen/rejected pair shares one label-specific alignment context.
"""

from __future__ import annotations


SYSTEM_PROMPT = """
You are generating spoken Cookie Theft picture descriptions as if they were transcript excerpts from a human participant.

Your output must sound like natural speech, not like a written image caption, not like a polished summary, and not like an essay.

Priority order:
1. Match the speaking style of the ORIGINAL TARGET TEXT as closely as possible.
2. Keep the output natural as spoken language.
3. Then satisfy the required mode-specific constraints.

Important style rules:
- Write as if the speaker is looking at the picture and talking in real time.
- Follow the wording habits, sentence rhythm, hesitation pattern, and discourse style of the original target text.
- Prefer spoken phrasing over polished written narration.
- It is acceptable to use pauses, repairs, repetitions, fillers, loose syntax, and incomplete sentences if they fit the source style.
- Do not sound like a formal narrator, clinician, annotator, or AI assistant.
- Do not add abstract interpretations, morals, or scene summaries.

Strong prohibitions:
- Do NOT write phrases like:
  "In the image..."
  "In the scene..."
  "The scene portrays..."
  "The overall atmosphere..."
  "This depicts..."
  "Interestingly..."
  "Overall..."
- Do NOT add literary commentary or neat wrap-up sentences.
- Do NOT make the output sound cleaner, more elegant, or more report-like than the original target text unless the mode explicitly requires a violation.

We will evaluate the output along three dimensions:
- Coverage: how many information units from the image are mentioned.
- Depth: how specifically the mentioned units are described.
- Style Residual: how different the phrasing/style is from the source in residual embedding space.

Two constraints are used:
1) Cognitive Consistency
2) Style Diversity Expansion

You must follow the required MODE exactly:
- MODE=CHOSEN: satisfy both constraints as much as possible.
- MODE=REJECTED: intentionally violate at least one required constraint, while still sounding like plausible human speech.

Bad output characteristics to avoid:
- polished image caption style
- formal written narration
- summary-first structure
- moral interpretation
- atmosphere commentary
- phrases such as "in the image", "in this scene", "overall", "the scene depicts", "interestingly"
""".strip()


AD_ALIGNMENT_INSTRUCTION = """
MODE=CHOSEN.

Generate a spoken Cookie Theft description that sounds like an Alzheimer's Disease (AD) participant.

Target style:
- Follow the ORIGINAL TARGET TEXT closely in speech texture.
- Keep the language oral, immediate, and transcript-like.
- Preserve AD-like characteristics when they are present in the source style: hesitations, repetitions, empty words, fragmentation, self-correction, reduced specificity, or loosened structure.
- Do not become noticeably more coherent, more detailed, or more information-dense than the source.

Constraint goals:
- Keep Coverage and Depth close to the source AD text.
- Do NOT increase Coverage too much.
- Do NOT increase Depth too much.
- Increase style diversity by changing wording, phrasing, and local structure.
- Do NOT copy sentence templates from the source.

Most important:
- It must sound like a person speaking, not like a rewritten visual description.
- Stay close to the source speaker's way of talking.
""".strip()


HC_ALIGNMENT_INSTRUCTION = """
MODE=CHOSEN.

Generate a spoken Cookie Theft description that sounds like a Healthy Control (HC) participant.

Target style:
- Follow the ORIGINAL TARGET TEXT closely in speech texture, rhythm, and discourse pattern.
- Keep the description coherent and informative, but still spoken and transcript-like.
- The output may be fluent and relatively complete, but it must not sound like a formal written caption or polished summary.

Constraint goals:
- Keep Coverage and Depth close to the source HC text.
- Do NOT drop Coverage too much.
- Do NOT drop Depth too much.
- Increase style diversity by varying wording, phrasing, sentence shape, and ordering.
- Do NOT copy sentence templates from the source.

Most important:
- Sound like a real participant describing the picture aloud.
- Do not sound like an AI-generated explanation of the image.
""".strip()


ORAL_STYLE_REMINDER = """
Speech-style reminder:
- Sound like spontaneous spoken description.
- Prefer simple discourse markers such as "and", "uh", "well", "here", "there", "I guess", "looks like", if they fit the source style.
- You may use repetitions, false starts, repairs, or loose phrasing if they match the original.
- Avoid polished transitions and avoid elegant summary statements.
- Do not invent a narrator voice.
- Do not explain the picture as if writing for a reader.
""".strip()


PROMPT_TEMPLATE = """{system_prompt}
{instruction}
{oral_style_reminder}
The original target text is:
[{original_text}]

Now produce ONE new Cookie Theft description.
It must sound like spoken transcript language in the style of the original target text.
Do not explain your choices.
Do not summarize the scene in a formal way.
Only output the simulated transcript."""


def build_alignment_task(label: int, original_text: str) -> str:
    """Return the shared chosen/rejected conditioning context for one pair."""

    if label not in {0, 1}:
        raise ValueError(f"Expected binary label 0/1; received {label!r}")
    source = str(original_text).strip()
    if not source:
        raise ValueError("The source transcript is empty.")
    instruction = AD_ALIGNMENT_INSTRUCTION if label == 1 else HC_ALIGNMENT_INSTRUCTION
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        instruction=instruction,
        oral_style_reminder=ORAL_STYLE_REMINDER,
        original_text=source,
    ).strip()

