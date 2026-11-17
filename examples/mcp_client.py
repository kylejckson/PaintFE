#!/usr/bin/env python3
"""Inspect PaintFE over MCP stdio, or draw three dots with --draw.

Uses only Python's standard library. Open the desktop editor separately first.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


class MCPError(RuntimeError):
    """A connection, protocol or tool execution error."""


class Client:
    def __init__(self, command, args=("--mcp",), timeout=20):
        self.timeout = timeout
        self.next_id = 0
        self.messages = queue.Queue()
        self.errors = deque(maxlen=16)
        self.process = subprocess.Popen(
            [str(command), *args], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline(8 * 1024 * 1024 + 1)
                if not line:
                    raise MCPError("The MCP bridge closed its output stream.")
                if len(line) > 8 * 1024 * 1024:
                    raise MCPError("The MCP response exceeds the example's size limit.")
                message = json.loads(line.decode("utf-8"))
                if not isinstance(message, dict):
                    raise MCPError("Expected a JSON-RPC message object.")
                self.messages.put(message)
        except (ValueError, OSError, MCPError) as error:
            self.messages.put(MCPError(str(error)))

    def _read_stderr(self):
        try:
            for line in self.process.stderr:
                self.errors.append(line.decode("utf-8", errors="replace").strip()[:2048])
        except OSError:
            pass

    def send(self, message):
        # MCP stdio is one UTF-8 JSON message per line, without headers.
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        try:
            self.process.stdin.write(payload.encode("utf-8") + b"\n")
            self.process.stdin.flush()
        except OSError as error:
            raise MCPError("Could not write to the MCP bridge.") from error

    def request(self, method, params=None):
        self.next_id += 1
        request_id = self.next_id
        self.send({"jsonrpc": "2.0", "id": request_id,
                   "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"Timed out waiting for {method}; check the client framing and bridge process.")
            try:
                response = self.messages.get(timeout=remaining)
            except queue.Empty as error:
                raise MCPError(f"Timed out waiting for {method}; check the client framing and bridge process.") from error
            if isinstance(response, Exception):
                detail = "\n".join(self.errors)
                raise MCPError(str(response) + ("\n" + detail if detail else ""))
            if response.get("id") != request_id:
                continue  # Notifications do not complete a pending request.
            if "error" in response:
                raise MCPError(str(response["error"].get("message", response["error"])))
            if "result" not in response:
                raise MCPError("The JSON-RPC response has no result.")
            return response["result"]

    def connect(self):
        result = self.request("initialize", {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "paintfe-python-example", "version": "1.0.0"},
        })
        if result.get("protocolVersion") != "2025-11-25":
            raise MCPError("The server did not accept the requested protocol version.")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return self.request("tools/list")

    def tool(self, name, arguments=None):
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        text = "\n".join(item["text"] for item in result.get("content", [])
                         if item.get("type") == "text")
        if result.get("isError"):
            raise MCPError(f"{name}: {text or 'tool execution failed'}")
        if "structuredContent" in result:
            return result["structuredContent"]
        try:
            return json.loads(text)
        except ValueError as error:
            raise MCPError(f"{name}: expected a JSON text result.") from error

    def wait_for_job(self, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.tool("job_status")
            if not isinstance(result, dict) or not isinstance(result.get("busy"), bool):
                raise MCPError("job_status: missing boolean busy field.")
            if not result["busy"]:
                if result.get("status") != "Applied":
                    raise MCPError("Drawing did not complete: " + str(result.get("status") or "no job result"))
                return result
            time.sleep(0.1)
        raise MCPError("Drawing is still running. Inspect job_status or use cancel before continuing.")

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=3)
        finally:
            self.process.stdout.close()
            self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def current_document(client, document_id=None):
    documents = client.tool("documents")
    if not isinstance(documents, list):
        raise MCPError("documents: expected a list.")
    for document in documents:
        if document_id is None or document.get("document") == document_id:
            return document
    raise MCPError("Open a document in PaintFE before continuing.")


def target(document):
    return {"document": document["document"], "layer": document["active_layer"],
            "revision": document["revision"]}


def draw_three_dots(client, document):
    document_id = document["document"]
    client.tool("new_layer", {**target(document), "name": "MCP example"})
    for fraction, colour in [(0.2, [255, 0, 0, 255]), (0.5, [0, 180, 0, 255]),
                             (0.8, [0, 80, 255, 255])]:
        document = current_document(client, document_id)
        x = min(document["width"] - 1, int(document["width"] * fraction))
        y = min(document["height"] - 1, int(document["height"] * fraction))
        result = client.tool("stroke", {**target(document), "points": [[x, y]],
                                       "colour": colour, "size": 3, "pencil": True})
        if result.get("queued") is not True:
            raise MCPError("stroke: expected a queued operation.")
        client.wait_for_job()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paintfe", required=True, type=Path, help="Path to the PaintFE executable")
    parser.add_argument("--draw", action="store_true", help="Create a layer and draw three test dots")
    parser.add_argument("--timeout", type=float, default=20, help="Response timeout in seconds (default: 20)")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
        parser.error("--timeout must be greater than zero and at most 120 seconds")
    try:
        with Client(args.paintfe, timeout=args.timeout) as client:
            tools = client.connect()
            print(f"Connected. {len(tools['tools'])} tools available.")
            document = current_document(client)
            print(f"Document: {document['name']} ({document['width']} x {document['height']})")
            if args.draw:
                draw_three_dots(client, document)
                print("Three dots added to the MCP example layer. Review in PaintFE; nothing was saved.")
            else:
                print("Read-only check complete. Use --draw to add a separate layer with three test dots.")
        return 0
    except (OSError, MCPError) as error:
        print(f"MCP example: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
