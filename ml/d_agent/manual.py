import subprocess # Applescript
# import pyautogui


# Example 
def execute_command(script):
    subprocess.run(["osascript", "-e", script])

script = '''
tell application "Mail"
        set selected_drafts to selection
        if (count of selected_drafts) is 0 then
                display dialog "No draft selected. Please select a draft to edit."
                return
        end if
        set the_draft to item 1 of selected_drafts
        tell the_draft
                set to recipients to "thekiwalkars@gmail.com"
        end tell
        activate
end tell
'''

subprocess.run(["osascript", "-e", script])