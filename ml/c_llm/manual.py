import subprocess # Applescript
# import pyautogui


# Example 
def execute_command(script):
    subprocess.run(["osascript", "-e", script])

script = '''
tell application "Mail"
	activate
	try
		-- Targets the draft window currently in focus
		set theMessage to outgoing message 1
		set content of theMessage to "today is january 5th 2026"
	on error
		display dialog "No draft window found."
	end try
end tell
'''

subprocess.run(["osascript", "-e", script])