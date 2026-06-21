import subprocess
import os

result = subprocess.run(['git', 'log', '--reverse', '--format=%H||%s'], capture_output=True, text=True)
commits = result.stdout.strip().split('\n')

replacements = {
    "fix: use absolute paths resolved from __file__": "fixed a bug where the models folder couldn't be found",
    "chore: reorganize directory structure and update README": "cleaned up the folders and made the README look amazing",
    "chore: remove obsolete project guide": "deleted the old guide since we have a better readme now",
    "feat: implement web-based interactive data collection dashboard with Flask and WebSockets": "built the brand new web UI for collecting eye data",
    "Implement core eye tracking and OS control logic": "added the main eye tracking and mouse control logic",
    "Add training script for the gaze estimation model": "added the AI model training script",
    "Create data collection script for dataset generation": "created the original data collection script",
    "Add project documentation and roadmap": "uploaded the project roadmap pdf",
    "Add MediaPipe face landmarker model": "uploaded the mediapipe AI model weights",
    "Initial project setup with ignore rules and README": "started the awesome eye tracking project!"
}

# Ensure we are on main
subprocess.run(['git', 'checkout', 'main'])
subprocess.run(['git', 'branch', '-D', 'new_main']) # delete if exists
subprocess.run(['git', 'checkout', '--orphan', 'new_main'])
subprocess.run(['git', 'rm', '-rf', '.'])

for i, line in enumerate(commits):
    if not line: continue
    h, msg = line.split('||', 1)
    new_msg = msg
    for k, v in replacements.items():
        if k in new_msg:
            new_msg = new_msg.replace(k, v)
    
    if i == 0:
        subprocess.run(['git', 'checkout', h, '--', '.'])
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', new_msg])
    else:
        subprocess.run(['git', 'cherry-pick', '-n', h])
        subprocess.run(['git', 'commit', '-m', new_msg])

subprocess.run(['git', 'branch', '-M', 'new_main', 'main'])
subprocess.run(['git', 'push', '--force', 'origin', 'main'])
