#!/usr/bin/env bash
# Open macOS System Settings panes for Jarvis permissions.
set -euo pipefail

echo "Jarvis requires the following macOS permissions:"
echo "  1. Microphone — for voice input"
echo "  2. Automation — Mail, Calendar, Notes, System Events"
echo "  3. Accessibility — UI automation fallback"
echo ""
echo "Opening System Settings..."

open "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone" 2>/dev/null || \
  open "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Microphone" 2>/dev/null || true

sleep 1
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation" 2>/dev/null || true

sleep 1
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility" 2>/dev/null || true

echo ""
echo "Grant access to Terminal (or your IDE) and enable:"
echo "  - Mail, Calendar, Notes, System Events"
echo ""
echo "Then run: jarvis health"
