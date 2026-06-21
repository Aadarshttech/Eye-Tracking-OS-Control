import subprocess
import os

if os.path.exists('vc_redist.x64.exe'):
    os.remove('vc_redist.x64.exe')
if os.path.exists('replay_commits.py'):
    os.remove('replay_commits.py')
if os.path.exists('rewrite.sh'):
    os.remove('rewrite.sh')

result = subprocess.run(['git', 'log', '--reverse', '--format=%H||%s'], capture_output=True, text=True)
commits = result.stdout.strip().split('\n')

replacements = {
    "fixed a bug where the models folder couldn't be found": "Fix relative paths to absolute paths",
    "cleaned up the folders and made the README look amazing": "Reorganize project directory structure and update README",
    "deleted the old guide since we have a better readme now": "Remove obsolete project guide",
    "built the brand new web UI for collecting eye data": "Implement web-based data collection dashboard",
    "added the main eye tracking and mouse control logic": "Implement core eye tracking and OS control logic",
    "added the AI model training script": "Add gaze model training script",
    "created the original data collection script": "Add data collection script",
    "uploaded the project roadmap pdf": "Add project roadmap documentation",
    "uploaded the mediapipe AI model weights": "Add face landmarker model",
    "started the awesome eye tracking project!": "Initial project setup"
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
        subprocess.run(['git', 'rm', '-rf', '--cached', '--ignore-unmatch', 'vc_redist.x64.exe', 'replay_commits.py', 'clean_history.py', 'rewrite.sh', 'rewrite_msg.py', 'edit_rebase.py', 'edit_msg.py'])
        subprocess.run(['git', 'add', '.'])
        subprocess.run(['git', 'commit', '-m', new_msg])
    else:
        subprocess.run(['git', 'cherry-pick', '-n', h])
        subprocess.run(['git', 'rm', '-rf', '--cached', '--ignore-unmatch', 'vc_redist.x64.exe', 'replay_commits.py', 'clean_history.py', 'rewrite.sh', 'rewrite_msg.py', 'edit_rebase.py', 'edit_msg.py'])
        subprocess.run(['git', 'commit', '-m', new_msg])

subprocess.run(['git', 'branch', '-M', 'new_main', 'main'])
subprocess.run(['git', 'push', '--force', 'origin', 'main'])
