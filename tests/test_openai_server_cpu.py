import concurrent.futures
import queue
import threading

import numpy as np

from openai_server import OpenAIServer, Result


def test_completion_batch_enqueues_before_waiting_and_preserves_order():
    server = OpenAIServer.__new__(OpenAIServer)
    server.jobs = queue.Queue()
    server.stopping = threading.Event()
    server.served = 0
    server.chat_mode = "response"
    server.default_max_tokens = 4
    server.default_temperature = 0.0
    server.default_top_p = 1.0
    server.model_name = "test"
    server._prompt_tokens = lambda ids: np.asarray(ids, np.int32)
    server._decode = lambda ids: " ".join(str(t) for t in ids)
    prompts = [[i, i + 10] for i in range(8)]
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(server.completions, {"prompt": prompts})
        try:
            for _ in prompts:
                jobs.append(server.next_job(timeout=2))
                assert jobs[-1] is not None
            for i in reversed(range(len(jobs))):
                np.testing.assert_array_equal(jobs[i].tokens, prompts[i])
                jobs[i].complete(Result(generated=[100 + i], finish_reason="length"))
            result = response.result(timeout=2)
        finally:
            for job in jobs:
                if job is not None and not job.done.is_set():
                    job.fail("test cleanup")
    assert [choice["index"] for choice in result["choices"]] == list(range(8))
    assert [choice["text"] for choice in result["choices"]] == [str(100 + i) for i in range(8)]
    assert result["usage"] == {"prompt_tokens": 16, "completion_tokens": 8, "total_tokens": 24}
    assert server.served == 8
