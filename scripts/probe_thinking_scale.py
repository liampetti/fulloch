"""Run the reviewed thirty-source regression scaffold with an explicit real model.

Requires a repository checkout including tests/fixtures. Sources are synthetic;
no fresh web retrieval, vault writes, household actions or application deployment.
This artifact reports failures, not fresh-report or ASR/TTS-active acceptance.
"""

import argparse
import json
import os
import time
from pathlib import Path

from tests.test_thinking_scale_run import scale_job, scale_worker

from core.background_jobs import BackgroundJob
from core.slm import generate_slm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["THINKING_SOURCE_STORE_DIR"] = str(args.output.parent / "scale-originals")
    from core.llm_openai import OpenAIClient

    client = OpenAIClient(args.model, args.base_url, "scale-local")
    client.context_tokens = 12_288

    class RealModel:
        context_tokens = 12_288
        _fulloch_remote = True
        _client = client._client
        model = args.model
        count_message_tokens = client.count_message_tokens

        def generate(self, _job, **kwargs):
            return generate_slm(client, **kwargs)

    job, result = scale_job(), {"mode": "synthetic_batches_real_model", "fresh_retrieval": False,
                               "asr_tts_active": False, "application_deployed": False,
                               "configuration": {"model": args.model, "base_url": args.base_url,
                                                 "request_context_tokens": 12_288},
                               "review": "Check accepted statements and analytical premises against retained originals."}
    started = time.monotonic()
    try:
        service, cancelled, snapshots = scale_worker(job, RealModel())
        report, _ = service.run(job, cancelled)
        result.update(first_run=job.checkpoint(), first_report=report, checkpoints=len(snapshots),
                      paused_before_final_batch=cancelled())
        if cancelled():
            job = BackgroundJob.from_checkpoint(job.checkpoint())
            service, cancelled, _ = scale_worker(job, RealModel(), pause=False)
            report, _ = service.run(job, cancelled)
            result.update(final_report=report, resumed=job.checkpoint())
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", interrupted=job.checkpoint())
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        client._client.close()


if __name__ == "__main__":
    main()
