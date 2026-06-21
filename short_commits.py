import subprocess
import os

result = subprocess.run(['git', 'log', '--reverse', '--format=%H||%s'], capture_output=True, text=True)
commits = result.stdout.strip().split('\n')

replacements = {
    "Fix relative paths to absolute paths": "fix: absolute paths in src/",
    "Reorganize project directory structure and update README": "refactor: move files to src/ and models/",
    "Remove obsolete project guide": "rm old guide",
    "Implement web-based data collection dashboard": "add src/web_collector.py",
    "Implement core eye tracking and OS control logic": "add src/eye_tracker.py",
    "Add gaze model training script": "add src/train_gaze_model.py",
    "Add data collection script": "add src/data_collector.py",
    "Add project roadmap documentation": "add docs/roadmap.pdf",
    "Add face landmarker model": "add models/face_landmarker.task",
    "Initial project setup": "init: base setup"
}

subprocess.run(['git', 'checkout', 'main'])
subprocess.run(['git', 'branch', '-D', 'new_main'])

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
