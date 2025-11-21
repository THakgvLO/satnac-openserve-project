"""
Run-all automation for the satnac-openserve project (Windows-friendly).

Usage:
    python run_all.py                # runs prepare_dashboard_data.py then ml_model.py
    python run_all.py --serve --open-browser    # then starts local server and opens dashboard
    python run_all.py --skip-prepare --skip-ml  # skip steps as needed
"""
import os
import sys
import subprocess
import argparse
import time
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Thread

ROOT = os.path.dirname(os.path.abspath(__file__))

def run_script(name, args=None):
    args = args or []
    exe = sys.executable
    cmd = [exe, name] + args
    print(f"Running: {' '.join(cmd)} (cwd={ROOT})")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Script {name} exited with code {res.returncode}")
    print(f"Finished: {name}")

def serve_folder(port=8000):
    os.chdir(ROOT)
    handler = SimpleHTTPRequestHandler
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"Serving {ROOT} at http://localhost:{port}/ (CTRL+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        httpd.server_close()

def file_exists(name):
    return os.path.exists(os.path.join(ROOT, name))

def main():
    parser = argparse.ArgumentParser(description="Run pipeline: prepare -> ml -> serve dashboard")
    parser.add_argument("--serve", action="store_true", help="Start local HTTP server after running scripts")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server (default: 8000)")
    parser.add_argument("--open-browser", action="store_true", help="Open default browser to dashboard.html after serving")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip prepare_dashboard_data.py step")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ml_model.py step")
    args = parser.parse_args()

    try:
        if not args.skip_prepare:
            if not file_exists("prepare_dashboard_data.py"):
                print("prepare_dashboard_data.py not found in project root, skipping prepare step.")
            else:
                run_script("prepare_dashboard_data.py")

        if not args.skip_ml:
            if not file_exists("ml_model.py"):
                print("ml_model.py not found in project root, skipping ml step.")
            else:
                run_script("ml_model.py")

        print("\nPipeline steps completed.")

        # Sanity checks (non-blocking)
        for expected in ("categorized_sites_output.csv", "dashboard_data.json"):
            path = os.path.join(ROOT, expected)
            print(f" - {expected}: {'FOUND' if os.path.exists(path) else 'MISSING'}")

        if args.serve:
            server_thread = Thread(target=serve_folder, kwargs={"port": args.port}, daemon=True)
            server_thread.start()
            # allow server to start
            time.sleep(1)
            url = f"http://localhost:{args.port}/dashboard.html"
            if args.open_browser:
                webbrowser.open(url)
            # keep main thread alive while server runs
            try:
                while server_thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()