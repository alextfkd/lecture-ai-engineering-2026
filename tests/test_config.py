"""ローカルの .env は、未設定の環境変数だけを補う。"""

import os

from app.config import _load_dotenv, _parse_dotenv


def test_parse_dotenv_keeps_values_and_drops_blanks():
    text = """
# comment
GEMINI_API_KEY="abc"
export AIE_MODEL='gemini-3.5-flash'
EMPTY=
NOT_A_PAIR
"""
    assert _parse_dotenv(text) == {
        "GEMINI_API_KEY": "abc",
        "AIE_MODEL": "gemini-3.5-flash",
    }


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=from-file\nAIE_MODEL=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.delenv("AIE_MODEL", raising=False)

    _load_dotenv(env_file)

    assert os.environ["GEMINI_API_KEY"] == "from-env"
    assert os.environ["AIE_MODEL"] == "from-file"


def test_load_dotenv_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("AIE_MODEL", raising=False)
    _load_dotenv(tmp_path / ".env")
    assert "AIE_MODEL" not in os.environ
