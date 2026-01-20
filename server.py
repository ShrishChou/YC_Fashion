from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for SharedArrayBuffer (gaussian splat renderer workers)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

if __name__ == "__main__":
    host, port = "127.0.0.1", 8000
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()
