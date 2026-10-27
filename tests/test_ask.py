"""入力トークン上限の判定。APIは呼ばない。"""

import pytest

from app.ask import TokenLimitExceeded, ensure_within_token_limit, parse_token_limit


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

    def count_tokens(self, *, model: str, contents: str) -> _Count:
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
