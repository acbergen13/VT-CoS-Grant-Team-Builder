"""Streamlit entrypoint: `streamlit run main.py`.

View switch lives in session state ("browser" | "chat") so the two views share
one session and the solicitation handoff object survives navigation (req 24).
"""

import streamlit as st

from app import browser, chat
from gt.state import init_session_state

st.set_page_config(page_title="VT CoS Grant Team Builder", layout="wide")


def main() -> None:
    init_session_state()
    if st.session_state.view == "chat":
        chat.render()
    else:
        browser.render()


if __name__ == "__main__":
    main()
