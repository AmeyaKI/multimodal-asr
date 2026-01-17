import subprocess # Applescript
# import pyautogui


# Example 
def execute_command(script):
    subprocess.run(["osascript", "-e", script])

script = """
tell application "Calendar"
    activate
    set tomorrowDate to (current date) + (24 * 60 * 60)
    set hours of tomorrowDate to 12
    set minutes of tomorrowDate to 0
    set seconds of tomorrowDate to 0
    tell calendar "akiwalkar@berkeley.edu"
        set newEvent to make new event with properties {summary:"Test Event", start date:tomorrowDate, end date:(tomorrowDate + (1 * hours))}
    end tell
    view calendar at (current date)
    delay 0.5
    show newEvent
end tell
"""
execute_command(script)
list_script = 'tell application "Calendar" to get name of every calendar'