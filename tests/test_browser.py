"""Stage 7 browser error-path tests — AppTest, fully offline.

The browser must surface grants.gov failures as user-facing messages (req 23)
and never replace the page with a traceback. These cover a failed refresh and
a failed detail fetch; the successful selection/handoff path is exercised by
`tests/test_chat.py` via the shared session-state contract.
"""

from streamlit.testing.v1 import AppTest

import app.browser as browser
from gt.grants.client import GrantsClientError
from gt.grants.store import GrantStore


def test_refresh_error_is_user_facing(monkeypatch, tmp_path):
    store = GrantStore(tmp_path / "grants")
    monkeypatch.setattr(browser, "default_store", lambda: store)

    def fail_refresh(client, store):
        raise GrantsClientError("grants.gov is not responding")

    monkeypatch.setattr(browser, "run_refresh", fail_refresh)
    at = AppTest.from_string(
        "import app.browser as browser\nbrowser.render()\n", default_timeout=10
    )
    at.run()
    at.button[0].click().run()  # Refresh solicitations

    assert not at.exception
    assert any("Refresh failed" in e.value for e in at.error)
    assert any("not responding" in e.value for e in at.error)


def test_detail_error_is_user_facing(monkeypatch):
    def fail_detail(client, store, opp_id):
        raise GrantsClientError("detail endpoint is not responding")

    monkeypatch.setattr(browser, "load_detail", fail_detail)
    at = AppTest.from_string(
        "import app.browser as browser\n"
        "browser._render_detail_and_handoff("
        "None, None, {'opp_id': '42', 'title': 'T'}, [])\n",
        default_timeout=10,
    )
    at.run()

    assert not at.exception
    assert any("Could not load details" in e.value for e in at.error)
    assert any("not responding" in e.value for e in at.error)


def test_empty_store_prompts_a_refresh(monkeypatch, tmp_path):
    store = GrantStore(tmp_path / "grants")
    monkeypatch.setattr(browser, "default_store", lambda: store)
    at = AppTest.from_string(
        "import app.browser as browser\nbrowser.render()\n", default_timeout=10
    )
    at.run()

    assert not at.exception
    assert any("No solicitations" in i.value for i in at.info)
