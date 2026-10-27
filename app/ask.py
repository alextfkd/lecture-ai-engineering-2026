"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    AIE_TOKEN_LIMIT=20 uv run python -m app.ask "LoRAとは何ですか？"
                                        # 1回きりで上限を超えたら、モデルは呼ばない
                                        # 会話で上限を超えたら、履歴を要約してから続ける

    uv run python -m app.ask --new "LoRAとは何ですか？"
    uv run python -m app.ask --session 1 "それの欠点は？"
                                        # セッション番号ごとに sessions/N.json へ保存する

--new も --session も付けない呼び出しは、履歴を持たない1回きりの質問。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class TokenLimitExceeded(Exception):
    """入力トークン数が AIE_TOKEN_LIMIT を超えた。質問への応答はまだ呼んでいない。"""

    def __init__(self, prompt_tokens: int, limit: int, *, after_compact: bool = False) -> None:
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        self.after_compact = after_compact
        super().__init__(f"入力 {prompt_tokens} > 上限 {limit}")


class CompactFailed(Exception):
    """要約が空で、会話を続けられなかった。"""


def parse_token_limit(raw: str | None = None) -> int | None:
    """環境変数 AIE_TOKEN_LIMIT を読む。未設定なら制限なし。"""
    if raw is None:
        raw = os.getenv("AIE_TOKEN_LIMIT", "")
    raw = raw.strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError:
        raise SystemExit(f"AIE_TOKEN_LIMIT は正の整数にしてください: {raw}") from None
    if limit <= 0:
        raise SystemExit(f"AIE_TOKEN_LIMIT は正の整数にしてください: {raw}")
    return limit


def ensure_within_token_limit(
    client: genai.Client,
    contents: str | list,
    limit: int,
) -> int:
    """入力トークン数を数え、上限以下ならその数を返す。超えていたら TokenLimitExceeded。"""
    counted = client.models.count_tokens(model=MODEL, contents=contents)
    prompt_tokens = counted.total_tokens or 0
    if prompt_tokens > limit:
        raise TokenLimitExceeded(prompt_tokens, limit)
    return prompt_tokens


def sessions_dir() -> Path:
    """セッションJSONの置き場。リポジトリ直下の sessions/。"""
    return Path(__file__).resolve().parents[1] / "sessions"


def resolve_mode(new: bool, session: int | None) -> str:
    """one-shot / new / continue のどれで実行するかを決める。"""
    if new and session is not None:
        raise SystemExit("--new と --session は同時に指定できません。")
    if session is not None and session < 1:
        raise SystemExit("--session は 1 以上の整数にしてください。")
    if new:
        return "new"
    if session is not None:
        return "continue"
    return "once"


def resolve_question(question: str | None, mode: str) -> str:
    """1回きりの質問だけ、引数を省略したときに既定の質問を使う。"""
    if question:
        return question
    if mode == "once":
        return DEFAULT_QUESTION
    raise SystemExit("会話のときは質問を引数で指定してください。")


def next_session_id(directory: Path | None = None) -> int:
    """既存の sessions/N.json の最大番号の次。ファイルがなければ 1。"""
    directory = sessions_dir() if directory is None else directory
    if not directory.is_dir():
        return 1
    numbers = [int(path.stem) for path in directory.glob("*.json") if path.stem.isdigit()]
    return max(numbers, default=0) + 1


def load_session(session_id: int, directory: Path | None = None) -> dict:
    """セッションJSONを読む。無い、壊れている、中の番号が違うときは終了する。"""
    directory = sessions_dir() if directory is None else directory
    path = directory / f"{session_id}.json"
    if not path.is_file():
        raise SystemExit(f"セッション {session_id} が見つかりません: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"セッション {session_id} のJSONを読めません: {e}") from None
    turns = data.get("turns") if isinstance(data, dict) else None
    if not isinstance(data, dict) or not isinstance(turns, list):
        raise SystemExit(f"セッション {session_id} の形式が不正です: {path}")
    if data.get("session") != session_id:
        raise SystemExit(
            f"セッション {session_id} のファイルに別の番号 {data.get('session')} が入っています。"
        )
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "model"}
            or not isinstance(turn.get("text"), str)
        ):
            raise SystemExit(f"セッション {session_id} の発言の形式が不正です: {path}")
    return data


def save_session(data: dict, directory: Path | None = None) -> Path:
    """セッションJSONを書く。書きかけのファイルが残らないよう、一時ファイルから置き換える。"""
    directory = sessions_dir() if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['session']}.json"
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


def new_session(session_id: int) -> dict:
    return {"session": session_id, "model": MODEL, "turns": []}


def record_exchange(
    session: dict,
    question: str,
    answer: str | None,
    user_tokens: int | None,
    usage: types.GenerateContentResponseUsageMetadata | None,
) -> None:
    """成功した1往復を履歴の末尾に足す。失敗した呼び出しでは呼ばない。"""
    session["turns"].append(
        {"role": "user", "text": question, "tokens": user_tokens}
    )
    model_turn: dict = {
        "role": "model",
        "text": answer or "",
        "tokens": (usage.candidates_token_count or 0) if usage is not None else None,
    }
    if usage is not None:
        model_turn["prompt_tokens"] = usage.prompt_token_count or 0
        thoughts = usage.thoughts_token_count or 0
        if thoughts:
            model_turn["thoughts_tokens"] = thoughts
    session["turns"].append(model_turn)
    session["model"] = MODEL


def turn_contents(turns: list[dict]) -> list[types.Content]:
    return [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in turns
    ]


def conversation_transcript(turns: list[dict]) -> str:
    """要約プロンプトに埋め込む、役割つきの会話本文。"""
    labels = {"user": "利用者", "model": "モデル"}
    return "\n".join(f"{labels[turn['role']]}: {turn['text']}" for turn in turns)


def summary_prompt(turns: list[dict], prompt_tokens: int, limit: int) -> str:
    """入力トークン数と上限を添えて、会話の要約を頼む。"""
    return (
        "次の会話を、この後も会話を続けるために要約してください。\n"
        f"いまの入力トークン数は {prompt_tokens} です。入力の上限は {limit} トークンです。\n"
        "要約は上限より短く、目安として上限の半分以内に収めてください。\n"
        "固有名詞、数字、結論、まだ答えていない質問は残してください。\n"
        "要約文だけを返してください。\n\n"
        f"会話履歴:\n{conversation_transcript(turns)}"
    )


SUMMARY_PREFACE = "これまでの会話は次の要約に圧縮されています。この内容を前提に続けてください。"


def summary_history(
    summary: str,
    usage: types.GenerateContentResponseUsageMetadata | None,
    source_prompt_tokens: int,
) -> list[dict]:
    """要約を、この後の会話の先頭に置く2発言にする。"""
    model_turn: dict = {
        "role": "model",
        "text": summary,
        "tokens": (usage.candidates_token_count or 0) if usage is not None else None,
        "compacted": True,
        "source_prompt_tokens": source_prompt_tokens,
    }
    if usage is not None and (usage.thoughts_token_count or 0):
        model_turn["thoughts_tokens"] = usage.thoughts_token_count
    return [
        {"role": "user", "text": SUMMARY_PREFACE, "tokens": None, "compacted": True},
        model_turn,
    ]


@dataclass
class AskResult:
    text: str | None
    usage: types.GenerateContentResponseUsageMetadata | None
    user_tokens: int | None
    history: list[dict] | None
    summary: str | None = None
    summary_usage: types.GenerateContentResponseUsageMetadata | None = None
    compacted_from: int | None = None


def question_contents(question: str, prior_turns: list[dict] | None) -> str | list:
    if prior_turns is None:
        return question
    return turn_contents([*prior_turns, {"role": "user", "text": question}])


def ask_with_client(
    client: genai.Client,
    question: str,
    *,
    token_limit: int | None,
    prior_turns: list[dict] | None,
    config: types.GenerateContentConfig | None,
) -> AskResult:
    """質問する。会話で入力が上限を超えたら、先に履歴を要約してから送る。"""
    contents = question_contents(question, prior_turns)
    history = prior_turns
    summary = None
    summary_usage = None
    compacted_from = None
    if token_limit is not None:
        try:
            ensure_within_token_limit(client, contents, token_limit)
        except TokenLimitExceeded as exceeded:
            if not prior_turns:
                raise
            summary, summary_usage = summarize_history(
                client, prior_turns, exceeded.prompt_tokens, token_limit, config
            )
            history = summary_history(summary, summary_usage, exceeded.prompt_tokens)
            contents = question_contents(question, history)
            compacted_from = exceeded.prompt_tokens
            try:
                ensure_within_token_limit(client, contents, token_limit)
            except TokenLimitExceeded as still_over:
                raise TokenLimitExceeded(
                    still_over.prompt_tokens,
                    still_over.limit,
                    after_compact=True,
                ) from None
    user_tokens = None
    if prior_turns is not None:
        counted = client.models.count_tokens(model=MODEL, contents=question)
        user_tokens = counted.total_tokens or 0
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    return AskResult(
        text=response.text,
        usage=response.usage_metadata,
        user_tokens=user_tokens,
        history=history,
        summary=summary,
        summary_usage=summary_usage,
        compacted_from=compacted_from,
    )


def summarize_history(
    client: genai.Client,
    turns: list[dict],
    prompt_tokens: int,
    limit: int,
    config: types.GenerateContentConfig | None,
) -> tuple[str, types.GenerateContentResponseUsageMetadata | None]:
    """上限チェックを通さず、履歴と要約用プロンプトだけをモデルに渡す。"""
    response = client.models.generate_content(
        model=MODEL,
        contents=summary_prompt(turns, prompt_tokens, limit),
        config=config,
    )
    summary = (response.text or "").strip()
    if not summary:
        raise CompactFailed("会話の要約が空だったため、続けられませんでした。")
    return summary, response.usage_metadata


def ask(
    question: str,
    *,
    token_limit: int | None = None,
    prior_turns: list[dict] | None = None,
) -> AskResult:
    """質問する。prior_turns が None なら1回きり。リストならその履歴の続きとして送る。"""
    client = genai.Client(api_key=get_api_key())
    raw_temperature = os.getenv("AIE_TEMPERATURE", "").strip()
    config = (
        types.GenerateContentConfig(temperature=float(raw_temperature))
        if raw_temperature
        else None
    )
    return ask_with_client(
        client,
        question,
        token_limit=token_limit,
        prior_turns=prior_turns,
        config=config,
    )


def format_usage(
    usage: types.GenerateContentResponseUsageMetadata | None,
    token_limit: int | None,
    *,
    label: str = "[tokens]",
) -> str | None:
    if usage is None:
        return None
    prompt = usage.prompt_token_count or 0
    output = usage.candidates_token_count or 0
    total = usage.total_token_count or 0
    thoughts = usage.thoughts_token_count or 0
    line = f"{label} 入力 {prompt} / 出力 {output}"
    if thoughts:
        line += f" / 思考 {thoughts}"
    line += f" / 合計 {total}"
    if token_limit is not None:
        line += f" / 上限 {token_limit}"
    return line


def print_usage(
    usage: types.GenerateContentResponseUsageMetadata | None,
    token_limit: int | None,
) -> None:
    line = format_usage(usage, token_limit)
    if line is not None:
        print(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--new", action="store_true", help="新しい会話セッションを始める")
    parser.add_argument("--session", type=int, metavar="N", help="続けるセッション番号")
    args = parser.parse_args()

    mode = resolve_mode(args.new, args.session)
    question = resolve_question(args.question, mode)
    session = None
    if mode == "new":
        session = new_session(next_session_id())
    elif mode == "continue":
        session = load_session(args.session)

    print(f"[model] {MODEL}")
    temperature = os.getenv("AIE_TEMPERATURE", "").strip()
    if temperature:
        print(f"[temperature] {temperature}")
    if session is not None:
        kind = "新規" if mode == "new" else f"続き（これまでの発言 {len(session['turns'])}）"
        print(f"[session] {session['session']} {kind}")
        saved_model = session.get("model")
        if mode == "continue" and saved_model and saved_model != MODEL:
            print(f"[session] 開始モデルは {saved_model}")
    token_limit = parse_token_limit()
    if token_limit is not None:
        print(f"[token_limit] 入力上限 {token_limit}")
    print(f"[質問] {question}\n")
    try:
        prior_turns = None if session is None else session["turns"]
        result = ask(
            question,
            token_limit=token_limit,
            prior_turns=prior_turns,
        )
    except TokenLimitExceeded as e:
        print(f"[tokens] 入力 {e.prompt_tokens} / 上限 {e.limit}")
        if e.after_compact:
            print("要約したあとも入力が上限を超えたため、質問には答えていません。")
        else:
            print("入力トークンが上限を超えたため、モデルは呼び出していません。")
        raise SystemExit(1)
    except CompactFailed as e:
        print(e)
        raise SystemExit(1)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)

    if result.summary is not None and session is not None:
        session["turns"] = result.history
        print(
            f"[compact] 入力 {result.compacted_from} > 上限 {token_limit} のため、"
            "会話を要約してから続けます"
        )
        compact_line = format_usage(result.summary_usage, None, label="[compact]")
        if compact_line is not None:
            print(compact_line)
        print(f"\n{result.summary}\n")
    print_usage(result.usage, token_limit)
    print(result.text)
    if session is not None:
        record_exchange(session, question, result.text, result.user_tokens, result.usage)
        path = save_session(session)
        saved = path.relative_to(sessions_dir().parent)
        print(f"\n[saved] {saved}")


if __name__ == "__main__":
    main()
