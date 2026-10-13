"""Generate an audited report from saved original evidence, with no retrieval.

Example: .venv/bin/python -m scripts.generate_thinking_report --evidence saved.evidence.json
         --container fulloch-ai --output /tmp/opencode/report.md
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from core.thinking_document_report import DocumentReportGenerator


class DockerModel:
    """Reach a loopback-only model without exposing ports or altering its server."""

    research_output_ceiling = 8192

    def __init__(self, container, base_url, model):
        self.container, self.base_url, self.model = container, base_url.rstrip("/"), model
        self.calls = []

    def _request(self, path, payload, timeout=180):
        script = """import json,sys,time,urllib.request
url,payload,timeout=json.load(sys.stdin)
request=urllib.request.Request(url,json.dumps(payload).encode(),{'Content-Type':'application/json'})
started=time.monotonic()
with urllib.request.urlopen(request,timeout=timeout) as response:
    if not payload.get('stream'):
        sys.stdout.write(response.read().decode())
    else:
        content=[]; usage={}; first=None; reason=None
        for line in response:
            line=line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event=json.loads(line[6:])
            if event.get('usage'): usage=event['usage']
            for choice in event.get('choices',[]):
                delta=choice.get('delta',{})
                if first is None and (delta.get('content') or delta.get('reasoning_content')):
                    first=(time.monotonic()-started)*1000
                content.append(delta.get('content') or '')
                reason=choice.get('finish_reason') or reason
        print(json.dumps({'choices':[{'message':{'content':''.join(content)},'finish_reason':reason}],
                          'usage':usage,'ttft_ms':first}))
"""
        result = subprocess.run(
            ["docker", "exec", "-i", self.container, "python", "-c", script],
            input=json.dumps([self.base_url.removesuffix("/v1") + path, payload, timeout]),
            capture_output=True, text=True, timeout=timeout + 20, check=True,
        )
        return json.loads(result.stdout)

    def count_message_tokens(self, messages, *, thinking_mode=False):
        templated = self._request("/apply-template", {
            "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": thinking_mode},
        })
        return len(self._request("/tokenize", {"content": templated["prompt"], "add_special": True})["tokens"])

    def generate(self, *, diagnostics=None, **kwargs):
        messages = [{"role": "system", "content": kwargs["system_prompt"]},
                    {"role": "user", "content": kwargs["user_prompt"]}]
        payload = {
            "model": self.model, "messages": messages, "max_tokens": kwargs["max_new_tokens"],
            "temperature": kwargs["temperature"], "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": kwargs.get("thinking_mode", False)},
        }
        if kwargs.get("response_schema"):
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "research_stage", "strict": False, "schema": kwargs["response_schema"],
            }}
        started = time.monotonic()
        result = self._request("/v1/chat/completions", payload, timeout=kwargs.get("generation_timeout") or 180)
        choice = result["choices"][0]
        if diagnostics is not None:
            diagnostics.update(result.get("usage", {}),
                               ttft_ms=result.get("ttft_ms"),
                               termination="truncated" if choice["finish_reason"] == "length" else "completed")
        response = choice["message"].get("content") or ""
        self.calls.append({"user_prompt": kwargs["user_prompt"], "response": response,
                           "elapsed_seconds": round(time.monotonic() - started, 3),
                           "usage": result.get("usage", {}), "finish_reason": choice["finish_reason"]})
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container", help="Docker container with a loopback-only llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--context", type=int, default=12_288)
    args = parser.parse_args()
    diagnostics_path = args.output.with_suffix(".generation.json")
    if args.evidence.resolve() in {args.output.resolve(), diagnostics_path.resolve()}:
        parser.error("Outputs must not overwrite the input evidence")
    original = args.evidence.read_bytes()
    evidence = json.loads(original)
    if args.container:
        model = DockerModel(args.container, args.base_url, args.model)
    else:
        from core.llm_openai import load_openai

        _, model = load_openai(model=args.model, base_url=args.base_url)
    generator = DocumentReportGenerator(model, context_tokens=args.context)
    result = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = generator.generate(evidence)
        args.output.write_text(result.pop("report"), encoding="utf-8")
        print(f"Report: {args.output}")
    finally:
        result.update(calls=generator.trace, fixture_sha256=hashlib.sha256(original).hexdigest(),
                      model=args.model, context_tokens=args.context)
        diagnostics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        assert args.evidence.read_bytes() == original, "Input evidence changed"
        print(f"Generation diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
