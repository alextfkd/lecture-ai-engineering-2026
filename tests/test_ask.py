"""入力トークン上限とセッションJSON。APIは呼ばない。"""

import pytest

from app.ask import (
    CompactFailed,
    TokenLimitExceeded,
    ask_with_client,
    ensure_within_token_limit,
    load_session,
    new_session,
    next_session_id,
    parse_token_limit,
    record_exchange,
    resolve_mode,
    resolve_question,
    save_session,
    summary_prompt,
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


def test_summary_prompt_includes_token_count_limit_and_transcript():
    turns = [
        {"role": "user", "text": "合言葉はみかん"},
        {"role": "model", "text": "覚えた"},
    ]
    prompt = summary_prompt(turns, 1819, 1000)
    assert "1819" in prompt
    assert "1000" in prompt
    assert "利用者: 合言葉はみかん" in prompt
    assert "モデル: 覚えた" in prompt


class _Usage:
    def __init__(self, prompt: int, output: int) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = output
        self.total_token_count = prompt + output
        self.thoughts_token_count = 0


class _Response:
    def __init__(self, text: str, prompt: int, output: int) -> None:
        self.text = text
        self.usage_metadata = _Usage(prompt, output)


class _ScriptedModels:
    def __init__(self, counts: list[int], responses: list[_Response]) -> None:
        self.counts = list(counts)
        self.responses = list(responses)
        self.generate_contents: list[object] = []

    def count_tokens(self, *, model: str, contents: object) -> _Count:
        return _Count(self.counts.pop(0))

    def generate_content(self, *, model: str, contents: object, config: object) -> _Response:
        self.generate_contents.append(contents)
        return self.responses.pop(0)


class _ScriptedClient:
    def __init__(self, counts: list[int], responses: list[_Response]) -> None:
        self.models = _ScriptedModels(counts, responses)


def test_conversation_over_the_limit_is_summarized_then_answered():
    prior = [
        {"role": "user", "text": "合言葉はみかん", "tokens": 10},
        {"role": "model", "text": "覚えた", "tokens": 5, "prompt_tokens": 10},
    ]
    client = _ScriptedClient(
        counts=[1819, 200, 8],
        responses=[
            _Response("合言葉はみかんだった。", 1800, 12),
            _Response("みかん", 200, 2),
        ],
    )
    result = ask_with_client(
        client,
        "合言葉は何？",
        token_limit=1000,
        prior_turns=prior,
        config=None,
    )
    assert result.text == "みかん"
    assert result.compacted_from == 1819
    assert result.summary == "合言葉はみかんだった。"
    assert result.user_tokens == 8
    assert prior[0]["text"] == "合言葉はみかん"
    assert result.history is not None
    assert result.history[1]["text"] == "合言葉はみかんだった。"
    assert result.history[1]["source_prompt_tokens"] == 1819
    assert result.history[1]["compacted"] is True
    summary_request = client.models.generate_contents[0]
    assert isinstance(summary_request, str)
    assert "1819" in summary_request
    assert "1000" in summary_request
    answer_request = client.models.generate_contents[1]
    assert answer_request[-1].parts[0].text == "合言葉は何？"
    assert "合言葉はみかんだった。" in answer_request[1].parts[0].text


def test_one_shot_over_the_limit_does_not_summarize():
    client = _ScriptedClient(counts=[50], responses=[])
    with pytest.raises(TokenLimitExceeded) as exc:
        ask_with_client(client, "hello", token_limit=20, prior_turns=None, config=None)
    assert exc.value.after_compact is False
    assert client.models.generate_contents == []


def test_summary_that_is_still_over_the_limit_does_not_answer():
    prior = [{"role": "user", "text": "長い", "tokens": 1}, {"role": "model", "text": "履歴", "tokens": 1}]
    client = _ScriptedClient(
        counts=[101, 150],
        responses=[_Response("まだ長い要約", 90, 40)],
    )
    with pytest.raises(TokenLimitExceeded) as exc:
        ask_with_client(client, "続き", token_limit=100, prior_turns=prior, config=None)
    assert exc.value.after_compact is True
    assert exc.value.prompt_tokens == 150
    assert len(client.models.generate_contents) == 1


def test_empty_summary_does_not_continue():
    prior = [{"role": "user", "text": "長い", "tokens": 1}, {"role": "model", "text": "履歴", "tokens": 1}]
    client = _ScriptedClient(counts=[100], responses=[_Response("  ", 90, 0)])
    with pytest.raises(CompactFailed):
        ask_with_client(client, "続き", token_limit=50, prior_turns=prior, config=None)
    assert len(client.models.generate_contents) == 1
