import sys
import json
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from silex.core.cognitive_loop import CognitiveLoop
from silex.utils.logger import setup_logger

log = setup_logger("silex.daemon")

# Global singleton states
_engine_loop: CognitiveLoop = None
_async_loop: asyncio.AbstractEventLoop = None

class SilexDaemonHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json({"status": "online", "engine": "silex"})
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        
        try:
            req_data = json.loads(body)
        except Exception:
            self.send_error(400, "Invalid JSON")
            return

        if parsed.path == "/cognitive":
            user_text = req_data.get("text", "")
            images = req_data.get("images", [])
            
            # Bridge the sync HTTP thread to the background Asyncio loop
            future = asyncio.run_coroutine_threadsafe(
                _engine_loop.process(user_text, images=images),
                _async_loop
            )
            try:
                result = future.result(timeout=300.0)
                # result is a CognitiveResponse pydantic model
                self._send_json(result.model_dump())
            except Exception as e:
                log.error(f"Cognitive execution failed: {e}")
                self.send_error(500, str(e))
                
        elif parsed.path == "/command":
            cmd_name = req_data.get("command", "")
            
            async def _execute_cmd():
                if cmd_name == "memories":
                    return await _engine_loop.get_all_memories()
                elif cmd_name == "goals":
                    return await _engine_loop.get_all_goals()
                elif cmd_name == "stats":
                    # Mock stats for now
                    return {"context_used": 0, "context_max": 0, "tokens_saved": 0}
                elif cmd_name == "graph":
                    return []
                return []
                
            future = asyncio.run_coroutine_threadsafe(_execute_cmd(), _async_loop)
            try:
                result = future.result(timeout=10.0)
                # Convert pydantic models if needed
                if isinstance(result, list) and len(result) > 0 and hasattr(result[0], "model_dump"):
                    result = [r.model_dump() for r in result]
                self._send_json({"data": result})
            except Exception as e:
                self.send_error(500, str(e))
                
        elif parsed.path == "/shutdown":
            self._send_json({"status": "shutting_down"})
            # Trigger safe shutdown
            asyncio.run_coroutine_threadsafe(self.server.shutdown_server(), _async_loop)
            
        else:
            self.send_error(404)

    def _send_json(self, data: dict):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def log_message(self, format, *args):
        # Mute standard BaseHTTPRequestHandler access logs
        pass

class SilexDaemon(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)

    async def shutdown_server(self):
        log.info("Committing state and shutting down Silex Daemon...")
        # Close DB connections if needed, though aiosqlite handles garbage collection decently.
        # But we must stop the http server from blocking.
        # shutdown() must run in a separate thread because it blocks until the server stops,
        # and we are inside an asyncio task right now, initiated by a request handler thread.
        # If we call shutdown() here, it might deadlock waiting for the shutdown request to finish.
        def kill_server():
            self.shutdown()
        threading.Thread(target=kill_server, daemon=True).start()

def _start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def _init_engine():
    global _engine_loop
    log.info("Initializing Causal Memory Engine...")
    _engine_loop = CognitiveLoop()
    await _engine_loop.startup()

def run_daemon(port=8080):
    global _async_loop
    
    # 1. Boot background async event loop
    _async_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_start_async_loop, args=(_async_loop,), daemon=True)
    t.start()
    
    # 2. Boot the Cognitive Engine
    future = asyncio.run_coroutine_threadsafe(_init_engine(), _async_loop)
    future.result() # Blocks until startup completes
    
    # 3. Boot the native HTTP server
    server = SilexDaemon(('127.0.0.1', port), SilexDaemonHandler)
    log.info(f"Silex Daemon listening on 127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _async_loop.call_soon_threadsafe(_async_loop.stop)
        t.join()

if __name__ == "__main__":
    run_daemon()
