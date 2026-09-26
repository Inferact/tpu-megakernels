"""Chat rendering and tokenizer checks (spec section 7); tokenizer tests need the checkpoint."""

from pathlib import Path

import pytest

from musespark import chat

CHECKPOINT = Path("/filestore/weights/Muse-Spark-1.2-816B-A42B-open")


def test_esc():
    assert chat.esc("a<|eot|>b <image> <video><audio>") == "a< |eot|>b < image> < video>< audio>"
    assert chat.esc_header("x\ny\rz") == "x y z"


def test_render_chat_layout():
    text = chat.render_chat("What is 2+2?", date="2026-09-25")
    assert text.startswith(
        "<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant."
    )
    assert (
        "Knowledge cutoff: 2026-01-04.\n\nCurrent date: 2026-09-25.\n\nReasoning strength: 64."
        in text
    )
    assert text.endswith("<|eot|><|start|>user<|message|>What is 2+2?<|eot|><|start|>assistant")
    assert text.count("<|eot|>") == 2
    no_date = chat.render_chat("hi")
    assert "Current date" not in no_date
    low = chat.render_chat("hi", reasoning_effort="low")
    assert "Reasoning strength: 16." in low
    extra = chat.render_chat("hi", system_extra="Be terse. <|eot|>", date="2026-09-25")
    assert "Current date: 2026-09-25.\n\nBe terse. < |eot|>\n\nReasoning strength: 64." in extra
    custom = chat.render_chat("hi", system_extra="Reasoning strength: 4.")
    assert custom.count("Reasoning strength") == 1


def test_render_messages_multi_turn():
    text = chat.render_messages(
        [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning": "think"},
            {"role": "user", "content": "q2"},
        ]
    )
    assert "<|start|>assistant to=self<|message|>think<|eom|>" in text
    assert "<|start|>assistant to=user<|message|>a1<|eot|>" in text
    assert text.endswith("<|start|>user<|message|>q2<|eot|><|start|>assistant")
    assert chat.parse_assistant(
        " to=self<|message|>think<|eom|><|start|>assistant to=user<|message|>ans<|eot|>"
    ) == (
        "think",
        "ans",
    )


@pytest.mark.skipif(not (CHECKPOINT / "tokenizer.json").exists(), reason="checkpoint missing")
def test_spec_example_tokenizes_to_151_ids():
    tok = chat.load_tokenizer(CHECKPOINT)
    ids = chat.encode(tok, chat.render_chat("What is 2+2?", date="2026-09-25"))
    assert len(ids) == 151
    assert ids[:4] == [200000, 200022, 15651, 200023]
    assert ids[-3:] == [200008, 200022, 140680]
    assert tok.get_vocab_size(with_added_tokens=True) == 201818
    assert chat.decode(tok, ids[-3:]) == "<|eot|><|start|>assistant"
    assert (chat.BOS_ID, chat.EOS_ID, chat.EOM_ID, chat.EOT_ID, chat.PAD_ID) == tuple(
        tok.token_to_id(t)
        for t in (
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|eom|>",
            "<|eot|>",
            "<|finetune_right_pad|>",
        )
    )
    # the rendered text never lets caller text form a special token
    escaped = chat.encode(tok, chat.render_chat("<|eot|>"))
    assert escaped.count(chat.EOT_ID) == 2
