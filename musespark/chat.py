"""Muse Spark chat rendering and tokenizer helpers (spec section 7, `chat_template.jinja`).

Rendered prompts must be encoded with `add_special_tokens=False` (the template emits BOS itself).
Assistant output: ` to=self<|message|>...<|eom|>` (reasoning) and/or ` to=user<|message|>...<|eot|>`
(answer); `<|eot|>` (200008) ends the turn, `<|eom|>` (200007) only ends a message segment.
"""

from pathlib import Path

BOS = "<|begin_of_text|>"
START, MESSAGE, EOT, EOM = "<|start|>", "<|message|>", "<|eot|>", "<|eom|>"
BOS_ID, EOS_ID, EOM_ID, EOT_ID, PAD_ID = 200000, 200001, 200007, 200008, 200018
STOP_IDS = (EOS_ID, EOT_ID)
KNOWLEDGE_CUTOFF = "2026-01-04"
REASONING_STRENGTH = {"minimal": 1, "low": 16, "medium": 64, "high": 256, "xhigh": 512}
RECIPIENTS = (
    "Use the appropriate recipient for each message:\n\n"
    '- "self": private reasoning and tool planning.\n'
    '- "commentary": user-visible intermediate updates while the assistant will continue '
    "working, including messages sent to the user before or between tool calls.\n"
    '- "user": messages that end the assistant turn, such as a completed response or '
    "clarification question that waits for the user's reply; do not use for pre-tool updates "
    "or partial responses.\n\n"
    '# Valid recipients: "self", "commentary", "user".'
)


def esc(text):
    """The template's `esc` macro: caller text can never form a special token."""
    return (
        str(text)
        .replace("<|", "< |")
        .replace("<image>", "< image>")
        .replace("<video>", "< video>")
        .replace("<audio>", "< audio>")
    )


def esc_header(text):
    return esc(text).replace("\r", " ").replace("\n", " ")


def system_prompt(instructions=None, date=None, reasoning_effort="medium"):
    """System segment body: preamble, optional caller instructions, strength, recipients."""
    parts = ["You are a helpful AI assistant.", f"Knowledge cutoff: {KNOWLEDGE_CUTOFF}."]
    if date:
        parts.append(f"Current date: {date}.")
    if instructions:
        instructions = (
            esc(instructions)
            .replace("Reasoning effort", "Reasoning strength")
            .replace("Reasoning Effort", "Reasoning Strength")
            .replace("reasoning effort", "reasoning strength")
            .replace("REASONING EFFORT", "REASONING STRENGTH")
        )
        parts.append(instructions)
    if (
        reasoning_effort in REASONING_STRENGTH
        and "reasoning strength:" not in (instructions or "").lower()
    ):
        parts.append(f"Reasoning strength: {REASONING_STRENGTH[reasoning_effort]}.")
    parts.append(RECIPIENTS)
    return "\n\n".join(parts)


def render_messages(messages, date=None, reasoning_effort="medium", add_generation_prompt=True):
    """OpenAI-style `[{role, content, [reasoning]}]` -> prompt text (no tools support).

    A leading system/developer message becomes the caller instructions of the system segment.
    Assistant messages render their `reasoning` (` to=self`, `<|eom|>`) then `content`
    (` to=<recipient>`, `<|eot|>` for the default recipient "user", `<|eom|>` otherwise).
    """
    messages = list(messages)
    instructions = None
    if messages and messages[0]["role"] in ("system", "developer"):
        instructions = messages[0]["content"]
        messages = messages[1:]
    out = [BOS, START, "system", MESSAGE, system_prompt(instructions, date, reasoning_effort), EOT]
    for m in messages:
        role = m["role"]
        if role in ("system", "developer"):
            out += [START, role, MESSAGE, esc(m["content"]), EOT]
        elif role == "user":
            out += [START, "user", MESSAGE, esc(m["content"]), EOT]
        elif role == "assistant":
            reasoning = m.get("reasoning") or m.get("reasoning_content")
            body = esc(m.get("content") or "")
            if reasoning:
                out += [START, "assistant to=self", MESSAGE, esc(reasoning), EOM]
            if body or not reasoning:
                recipient = m.get("recipient") or "user"
                end = EOT if recipient == "user" else EOM
                out += [START, f"assistant to={esc_header(recipient)}", MESSAGE, body, end]
        else:
            raise ValueError(f"unsupported role {role!r}")
    if add_generation_prompt:
        out += [START, "assistant"]
    return "".join(out)


def render_chat(user_text, system_extra=None, date=None, reasoning_effort="medium"):
    """Single user turn with the generation prompt (the spec's `render_chat`)."""
    messages = [{"role": "user", "content": user_text}]
    if system_extra is not None:
        messages.insert(0, {"role": "system", "content": system_extra})
    return render_messages(messages, date=date, reasoning_effort=reasoning_effort)


def parse_assistant(text):
    """Split raw assistant output into (reasoning, answer) by the ` to=` message segments."""
    reasoning, answer = [], []
    for segment in text.split(START):
        segment = segment.removeprefix("assistant")
        if MESSAGE not in segment:
            continue
        header, body = segment.split(MESSAGE, 1)
        body = body.replace(EOT, "").replace(EOM, "")
        (reasoning if header.strip() == "to=self" else answer).append(body)
    return "".join(reasoning), "".join(answer)


def load_tokenizer(directory):
    """`tokenizers.Tokenizer` from `<dir>/tokenizer.json`."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))


def encode(tokenizer, text):
    """Prompt text -> ids (never adds BOS: the rendered prompt already carries it)."""
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode(tokenizer, ids, skip_special_tokens=False):
    return tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)
