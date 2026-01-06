import subprocess # Applescript
# import pyautogui


# Example 
def execute_command(script):
    subprocess.run(["osascript", "-e", script])

script = '''
tell application "Reminders"
	activate
	delay 1 -- Gives you time to see the app open
	
	-- 1. Create the list
	set newList to make new list with properties {name:"Real-Time List"}
	
	-- 2. Force the app to show the new list (Visual Step)
	-- This isn't strictly necessary for the data, but it's how you "follow" it
	delay 0.5
	
	-- 3. Add a reminder and watch it appear
	tell newList
		make new reminder with properties {name:"Watching this happen..."}
		delay 1
		make new reminder with properties {name:"Second task added!"}
	end tell
end tell
'''

subprocess.run(["osascript", "-e", script])