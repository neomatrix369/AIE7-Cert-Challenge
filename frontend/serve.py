#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend
Usage: python3 serve.py [port]
Default port: 3000
"""

import http.server
import socketserver
import sys
import os

# Get port from command line or use default
port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

# Create server
with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
    print(f"🚀 Frontend server starting on http://localhost:{port}")
    print(f"📁 Serving files from: {os.getcwd()}")
    print(f"🌐 Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")