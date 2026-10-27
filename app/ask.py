"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    AIE_TOKEN_LIMIT=20 uv run python -m app.ask "LoRAとは何ですか？"
                                        # 入力が上限を超えたら、モデルは呼ばない

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
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class TokenLimitExceeded(Exception):
    """入力トークン数が AIE_TOKEN_LIMIT を超えた。モデルはまだ呼んでいない。"""

    def __init__(self, prompt_tokens: int, limit: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        super().__init__(f"入力 {prompt_tokens} > 上限 {limit}")


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


def ask(
    question: str,
    *,
    token_limit: int | None = None,
    prior_turns: list[dict] | None = None,
) -> tuple[str | None, types.GenerateContentResponseUsageMetadata | None, int | None]:
    """質問する。prior_turns が None なら1回きり。リストならその履歴の続きとして送る。

    戻り値の3つ目は、この質問文だけのトークン数。会話の記録用で、1回きりのときは None。
    """
    client = genai.Client(api_key=get_api_key())
    raw_temperature = os.getenv("AIE_TEMPERATURE", "").strip()
    config = (
        types.GenerateContentConfig(temperature=float(raw_temperature))
        if raw_temperature
        else None
    )
    contents: str | list = (
        question if prior_turns is None else turn_contents([*prior_turns, {"role": "user", "text": question}])
    )
    if token_limit is not None:
        ensure_within_token_limit(client, contents, token_limit)
    user_tokens = None
    if prior_turns is not None:
        counted = client.models.count_tokens(model=MODEL, contents=question)
        user_tokens = counted.total_tokens or 0
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)
    return response.text, response.usage_metadata, user_tokens


def print_usage(
    usage: types.GenerateContentResponseUsageMetadata | None,
    token_limit: int | None,
) -> None:
    if usage is None:
        return
    prompt = usage.prompt_token_count or 0
    output = usage.candidates_token_count or 0
    total = usage.total_token_count or 0
    thoughts = usage.thoughts_token_count or 0
    line = f"[tokens] 入力 {prompt} / 出力 {output}"
    if thoughts:
        line += f" / 思考 {thoughts}"
    line += f" / 合計 {total}"
    if token_limit is not None:
        line += f" / 上限 {token_limit}"
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
        answer, usage, user_tokens = ask(
            question,
            token_limit=token_limit,
            prior_turns=prior_turns,
        )
    except TokenLimitExceeded as e:
        print(f"[tokens] 入力 {e.prompt_tokens} / 上限 {e.limit}")
        print("入力トークンが上限を超えたため、モデルは呼び出していません。")
        raise SystemExit(1)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)

    print_usage(usage, token_limit)
    print(answer)
    if session is not None:
        record_exchange(session, question, answer, user_tokens, usage)
        path = save_session(session)
        saved = path.relative_to(sessions_dir().parent)
        print(f"\n[saved] {saved}")


if __name__ == "__main__":
    main()
