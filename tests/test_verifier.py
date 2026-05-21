import os

os.environ["JARVIS_EVAL_MOCK"] = "1"

from jarvis.core.state import PlanStep
from jarvis.cognition.verifier import verify_step
from jarvis.tools import mail_automation


def test_verifier_rejects_wrong_subject():
    mail_automation.mail_compose()
    mail_automation.mail_set_subject("Wrong")
    step = PlanStep(
        tool="mail_set_subject",
        args={"subject": "Hello"},
        success_criteria="Hello",
        domain="mail",
    )
    ok, _ = verify_step(step, {"ok": True})
    assert not ok
