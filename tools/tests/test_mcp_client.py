"""Exercise the shipped example with real subprocess pipes and bounded replies."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("paintfe_mcp_example", ROOT / "examples/mcp_client.py")
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

# A small protocol fixture, not a replacement for the real executable tests.
SERVER = r'''
import json, sys
scenario = sys.argv[1]
polls = 0
for line in sys.stdin.buffer:
    assert line.endswith(b'\n') and not line.startswith(b'Content-Length:')
    message = json.loads(line.decode('utf-8'))
    if 'id' not in message:
        continue
    if scenario == 'silent':
        continue
    method = message['method']
    result = {}
    if method == 'initialize':
        result = {'protocolVersion':'2025-11-25'}
    elif method == 'tools/list':
        result = {'tools':[{'name':'documents'}]}
    elif method == 'tools/call':
        name = message['params']['name']
        if scenario == 'tool-error':
            result = {'isError':True,'content':[{'type':'text','text':'Document revision changed'}]}
        elif scenario == 'structured':
            result = {'content':[],'structuredContent':{'value':42}}
        elif name == 'job_status':
            polls += 1
            value = {'busy':polls == 1, 'status':'Processing' if polls == 1 else 'Applied'}
            if scenario == 'failed':
                value = {'busy':False,'status':'Document changed; result discarded'}
            if scenario == 'cancelled':
                value = {'busy':False,'status':'Cancelled'}
            if scenario == 'bad-status':
                value = {'status':'Applied'}
            result = {'content':[{'type':'text','text':json.dumps(value)}]}
        else:
            value = [{'document':'doc','name':'Café 日本語','active_layer':'top','revision':7,
                      'layers':[{'layer':'bottom'},{'layer':'top'}]}]
            result = {'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}]}
    if scenario == 'bad-output':
        sys.stdout.buffer.write(b'not JSON\n')
    else:
        if scenario == 'notification':
            note = {'jsonrpc':'2.0','method':'notifications/message','params':{}}
            sys.stdout.buffer.write(json.dumps(note).encode('utf-8')+b'\n')
        response = {'jsonrpc':'2.0','id':message['id'],'result':result}
        if scenario == 'rpc-error':
            response = {'jsonrpc':'2.0','id':message['id'],'error':{'code':-32603,'message':'No editor session'}}
        sys.stdout.buffer.write(json.dumps(response,ensure_ascii=False).encode('utf-8')+b'\n')
    sys.stdout.buffer.flush()
'''


class MCPClientTests(unittest.TestCase):
    def client(self, scenario="normal", timeout=5):
        return example.Client(sys.executable, ["-u", "-c", SERVER, scenario], timeout)

    def test_newline_handshake_and_unicode_reply_without_blank_lines(self):
        with self.client() as client:
            self.assertEqual(client.connect()["tools"][0]["name"], "documents")
            document = example.current_document(client)
            self.assertEqual(document["name"], "Café 日本語")
            self.assertEqual(example.target(document), {"document":"doc", "layer":"top", "revision":7})

    def test_notifications_do_not_complete_the_wrong_request(self):
        with self.client("notification") as client:
            self.assertEqual(client.connect()["tools"][0]["name"], "documents")

    def test_tool_execution_error_stops_the_client(self):
        with self.client("tool-error") as client:
            client.connect()
            with self.assertRaisesRegex(example.MCPError, "revision changed"):
                client.tool("stroke")

    def test_jsonrpc_error_is_reported(self):
        with self.client("rpc-error") as client:
            with self.assertRaisesRegex(example.MCPError, "No editor session"):
                client.connect()

    def test_busy_then_applied_is_success(self):
        with self.client() as client:
            client.connect()
            self.assertEqual(client.wait_for_job()["status"], "Applied")

    def test_failed_cancelled_and_missing_busy_do_not_look_successful(self):
        for scenario in ["failed", "cancelled", "bad-status"]:
            with self.subTest(scenario=scenario), self.client(scenario) as client:
                client.connect()
                with self.assertRaises(example.MCPError):
                    client.wait_for_job()

    def test_timeout_and_invalid_stdout_fail_without_hanging(self):
        with self.client("silent", timeout=0.2) as client:
            with self.assertRaisesRegex(example.MCPError, "Timed out"):
                client.connect()
        with self.client("bad-output") as client:
            with self.assertRaises(example.MCPError):
                client.connect()

    def test_structured_results_are_supported(self):
        with self.client("structured") as client:
            client.connect()
            self.assertEqual(client.tool("documents"), {"value":42})


if __name__ == "__main__":
    unittest.main()
