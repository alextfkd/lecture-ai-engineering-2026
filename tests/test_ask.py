"""入力トークン上限とセッションJSON。APIは呼ばない。"""

import pytest

from app.ask import (
    TokenLimitExceeded,
    ensure_within_token_limit,
    load_session,
    new_session,
    next_session_id,
    parse_token_limit,
    record_exchange,
    resolve_mode,
    resolve_question,
    save_session,
)


def test_parse_token_limit_unset_means_no_limit(monkeypatch):
    monkeypatch.delenv("AIE_TOKEN_LIMIT", raising=False)
    assert parse_token_limit() is None
    assert parse_token_limit("  ") is None


def test_parse_token_limit_accepts_positive_integer():
    assert parse_token_limit("20") == 20


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "abc"])
def test_parse_token_limit_rejects_non_positive_integer(raw):
    with pytest.raises(SystemExit):
        parse_token_limit(raw)


class _Count:
    def __init__(self, total_tokens: int | None) -> None:
        self.total_tokens = total_tokens


class _Models:
    def __init__(self, total_tokens: int | None) -> None:
        self.total_tokens = total_tokens

    def count_tokens(self, *, model: str, contents: object) -> _Count:
        self.model = model
        self.contents = contents
        return _Count(self.total_tokens)


class _Client:
    def __init__(self, total_tokens: int | None) -> None:
        self.models = _Models(total_tokens)


def test_limit_allows_input_equal_to_the_limit():
    assert ensure_within_token_limit(_Client(20), "hello", 20) == 20


def test_limit_blocks_when_input_exceeds_the_limit():
    with pytest.raises(TokenLimitExceeded) as exc:
        ensure_within_token_limit(_Client(21), "hello", 20)
    assert exc.value.prompt_tokens == 21
    assert exc.value.limit == 20


def test_limit_counts_the_contents_it_is_given():
    client = _Client(5)
    contents = ["history", "question"]
    ensure_within_token_limit(client, contents, 10)
    assert client.models.contents == contents


def test_resolve_mode_rejects_new_and_session_together():
    with pytest.raises(SystemExit):
        resolve_mode(True, 1)


def test_resolve_mode_rejects_non_positive_session():
    with pytest.raises(SystemExit):
        resolve_mode(False, 0)


def test_resolve_question_uses_default_only_for_one_shot():
    from app.ask import DEFAULT_QUESTION

    assert resolve_question(None, "once") == DEFAULT_QUESTION
    assert resolve_question("続きは？", "continue") == "続きは？"
    with pytest.raises(SystemExit):
        resolve_question(None, "new")


def test_session_json_roundtrip_assigns_the_next_number(tmp_path):
    assert next_session_id(tmp_path) == 1
    first = new_session(1)
    record_exchange(first, "合言葉はみかん", "覚えた", 4, None)
    save_session(first, tmp_path)
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    assert next_session_id(tmp_path) == 2
    loaded = load_session(1, tmp_path)
    assert loaded["turns"][0]["text"] == "合言葉はみかん"
    assert loaded["turns"][1]["role"] == "model"
    assert loaded["model"]


def test_load_session_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        load_session(3, tmp_path)
