from types import SimpleNamespace

import numpy as np
import pytest

from mpk.chat import MUSE_EOM, MUSE_MSG, MUSE_START, MuseTokenStreamer, _InteractiveStream, muse_split_channels
from openai_server import OpenAIServer, Result


class Tokenizer:
    text = {1: " to=self", 2: "Two plus two is four.", 3: "assistant to=user", 4: "4."}

    def decode(self, ids):
        return "".join(self.text.get(int(token), "") for token in ids)


@pytest.mark.parametrize("answer", (False, True))
def test_mpk_chat_channels_match_streaming_when_tokenizer_drops_markers(answer):
    tokenizer = Tokenizer()
    tokens = [1, MUSE_MSG, 2]
    if answer:
        tokens += [MUSE_EOM, MUSE_START, 3, MUSE_MSG, 4, 200008]
    streamer = MuseTokenStreamer(tokenizer)
    deltas = [delta for token in tokens for delta in streamer.feed([token])]
    expected = ("Two plus two is four.", "4." if answer else "")
    assert muse_split_channels(tokenizer, tokens) == expected
    assert "".join(text for channel, text in deltas if channel == "reasoning") == expected[0]
    assert "".join(text for channel, text in deltas if channel == "content") == expected[1]

    server = OpenAIServer.__new__(OpenAIServer)
    server.tokenizer = tokenizer
    server.model_name = "mpk-test"
    server.chat_mode = "response"
    server.max_prompt_tokens = 128
    server.chat_render = lambda *_: np.array([0], np.int32)
    server._params = lambda _: None
    server._job = lambda *args, **kwargs: SimpleNamespace(stop=[])
    server._run = lambda _: Result(generated=tokens, finish_reason="stop" if answer else "length")
    server.token_channel_splitter = lambda ids: muse_split_channels(tokenizer, ids)
    result = server.chat_completions({"messages": [{"role": "user", "content": "2+2?"}]})
    assert result["choices"][0]["message"] == {
        "role": "assistant", "content": expected[1], "reasoning_content": expected[0],
    }
    assert result["usage"]["completion_tokens"] == len(tokens)


@pytest.mark.parametrize("chat", (False, True))
def test_mpk_terminal_prints_raw_and_chat_output(chat, capsys):
    tokens = ([1, MUSE_MSG, 2, MUSE_EOM, MUSE_START, 3, MUSE_MSG, 4, 200008]
              if chat else [2, 4, 200008])
    stream = _InteractiveStream(Tokenizer(), chat=chat)
    try:
        for token in tokens:
            stream.submit([token])
    finally:
        stream.close()
    expected = ("[reasoning] Two plus two is four.\n\n[response]\n4."
                if chat else "Two plus two is four.4.")
    assert capsys.readouterr().out == expected
