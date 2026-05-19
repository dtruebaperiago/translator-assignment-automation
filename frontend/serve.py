"""
serve.py — iDISC AssignMate HUD local dev server
=================================================
Run this script from the frontend/ directory:

    python serve.py

It starts a local HTTP server on http://localhost:8080 and opens
idisc_hud.html in your default browser automatically.

No external dependencies — uses only Python's built-in http.server.
"""

import http.server
import socketserver
import webbrowser
import os
import threading

PORT = 8080
HOST = "localhost"

# Change working directory to the folder where this script lives
# (i.e. frontend/) so that relative paths like data/translators.json work.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class SilentHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with logging suppressed for cleaner output."""
    def log_message(self, format, *args):
        # Only log errors, not every request
        if args and len(args) >= 2 and str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def open_browser():
    url = f"http://{HOST}:{PORT}/idisc_hud.html"
    print(f"\n  Opening: {url}\n")
    webbrowser.open(url)


if __name__ == "__main__":
    with socketserver.TCPServer((HOST, PORT), SilentHandler) as httpd:
        httpd.allow_reuse_address = True
        print("=" * 55)
        print("  iDISC AssignMate HUD — Local Server")
        print("=" * 55)
        print(f"  Serving at : http://{HOST}:{PORT}/idisc_hud.html")
        print(f"  Folder     : {os.getcwd()}")
        print("  Press Ctrl+C to stop.\n")

        # Open browser after a short delay to let the server start
        threading.Timer(0.5, open_browser).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
