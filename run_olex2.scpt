-- This AppleScript automates the process of adding hydrogen atoms to the test set of QM9 using Olex2.
tell application "olex2"
	activate
	delay 1 -- Wait for OLEX2 to become the active application
	
	tell application "System Events"
		repeat with i from 0 to 13082
			-- 13082
			-- Open the input file
			keystroke "reap i" & i & ".cif"
			key code 36 -- Simulates pressing Enter
			-- Wait for the command to execute
			key code 36
			
			-- Add hydrogens
			keystroke "hadd"
			key code 36
			key code 36
			
			-- Save the output file
			keystroke "file o/" & i & ".xyz"
			key code 36
		end repeat
	end tell
end tell
