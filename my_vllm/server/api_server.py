"""OpenAI-compatible API server (stdlib, no external deps)."""

import json
import asyncio
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

from my_vllm.engine.llm_engine import LLMEngine
from my_vllm.server.async_llm import AsyncLLM


class CompletionHandler(BaseHTTPRequestHandler):
    engine: LLMEngine = None
    async_llm: AsyncLLM = None

    def do_POST(self):
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 50)
        temperature = body.get("temperature", 0.0)

        if temperature > 0:
            self._json(400, {"error": "only greedy sampling supported"})
            return

        try:
            result = CompletionHandler.engine.generate(
                prompt, max_new_tokens=max_tokens
            )
            response = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": 0,
                "model": "my-vllm",
                "choices": [{
                    "text": result["text"],
                    "index": 0,
                    "finish_reason": (
                        "stop" if "STOPPED" in result["finish_reason"]
                        else "length"
                    ),
                }],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["output_tokens"],
                    "total_tokens": (
                        result["prompt_tokens"] + result["output_tokens"]
                    ),
                },
            }
            self._json(200, response)
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress access logs


def create_server(
    engine: LLMEngine,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> HTTPServer:
    CompletionHandler.engine = engine
    server = HTTPServer((host, port), CompletionHandler)
    return server


def run_server(
    engine: LLMEngine,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    server = create_server(engine, host, port)
    print(f"Server listening on http://{host}:{port}")
    print(f"  POST /v1/completions   — OpenAI-compatible completion")
    print(f"  GET  /health            — health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
