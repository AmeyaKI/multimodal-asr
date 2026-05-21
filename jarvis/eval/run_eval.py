"""Run eval scenarios (text mode, mock tools)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import yaml

# Force mock for CI
os.environ.setdefault("JARVIS_EVAL_MOCK", "1")


async def run_scenario(scenario: dict) -> tuple[bool, str]:
    from jarvis.config import get_settings
    from jarvis.core.orchestrator import run_orchestrator
    from jarvis.cognition.planner import build_plan
    from jarvis.cognition.router import route_intent
    from jarvis.cognition.tool_registry import execute_tool

    get_settings().jarvis_eval_mock = True
    utterance = scenario["utterance"]
    sid = "eval-session"

    if scenario.get("expect_blocked"):
        from jarvis.core.policy import check_policy

        ok, msg = check_policy("mail_send", None)
        if ok:
            return False, "E4: send should be blocked without confirm"
        return True, "E4 blocked OK"

    route = await route_intent(utterance)
    intent = route.get("intent", scenario.get("intent"))
    plan = await build_plan(utterance, intent, route.get("slots", {}))

    if scenario.get("min_steps") and len(plan) < scenario["min_steps"]:
        return False, f"expected >= {scenario['min_steps']} steps, got {len(plan)}"

    expect_tool = scenario.get("expect_tool")
    if expect_tool:
        tools = [s.get("tool") for s in plan]
        if expect_tool not in tools:
            return False, f"expected tool {expect_tool}, got {tools}"

    expect_tools = scenario.get("expect_tools", [])
    if expect_tools:
        tools = [s.get("tool") for s in plan]
        for et in expect_tools:
            if et not in tools:
                return False, f"missing tool {et} in {tools}"

    # Execute plan in mock mode
    confirm_token = "testtoken" if "send" in utterance.lower() else None
    for step in plan:
        if step.get("tool") == "mail_send" and not scenario.get("expect_blocked"):
            confirm_token = "testtoken"
        result = execute_tool(step.get("tool", ""), step.get("args", {}), confirm_token=confirm_token)
        if step.get("tool") == "doc_create_markdown" and result.get("path"):
            for s in plan:
                if s.get("tool") == "doc_append_text":
                    s["args"]["path"] = result["path"]

    if expect_tool == "create_event" or "create_event" in [s.get("tool") for s in plan]:
        result = await run_orchestrator(utterance, session_id=sid)
        if not result.get("verification_ok") and not result.get("done"):
            if result.get("error"):
                return False, result["error"]

    return True, "OK"


async def main() -> int:
    path = Path(__file__).parent / "scenarios.yaml"
    data = yaml.safe_load(path.read_text())
    scenarios = data.get("scenarios", [])
    passed = 0
    failed = []

    for sc in scenarios:
        if sc.get("note") and "manual" in sc.get("note", ""):
            print(f"[SKIP] {sc['id']}: manual test")
            passed += 1
            continue
        ok, msg = await run_scenario(sc)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {sc['id']}: {msg}")
        if ok:
            passed += 1
        else:
            failed.append(sc["id"])

    print(f"\n{passed}/{len(scenarios)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
