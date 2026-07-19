import urllib.request
import subprocess
import os
import re

URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
EXE = "cloudflared.exe"

if not os.path.exists(EXE):
    print("Downloading Cloudflared...")
    urllib.request.urlretrieve(URL, EXE)
    print("Download complete.")

print("Starting Cloudflare Tunnel...")
# cloudflared prints logs to stderr
p = subprocess.Popen([EXE, "tunnel", "--url", "http://127.0.0.1:5000"], stderr=subprocess.PIPE, text=True)

print("\n--- WAITING FOR TUNNEL URL ---\n")
for line in p.stderr:
    if "trycloudflare.com" in line:
        match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
        if match:
            print("\n" + "="*60)
            print(" PUBLIC LINK GENERATED SUCCESSFULLY ")
            print("="*60)
            print(f"Give this exact link to ALL your friends:")
            print(f"{match.group(1)}")
            print("="*60)
            print("\n(Leave this window open to keep the link active)")
            break
