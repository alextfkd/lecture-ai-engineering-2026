"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    AIE_TOKEN_LIMIT=20 uv run python -m app.ask "LoRAとは何ですか？"
                                        # 入力が上限を超えたら、モデルは呼ばない

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os

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


def ensure_within_token_limit(client: genai.Client, contents: str, limit: int) -> int:
    """入力トークン数を数え、上限以下ならその数を返す。超えていたら TokenLimitExceeded。"""
    counted = client.models.count_tokens(model=MODEL, contents=contents)
    prompt_tokens = counted.total_tokens or 0
    if prompt_tokens > limit:
        raise TokenLimitExceeded(prompt_tokens, limit)
    return prompt_tokens


def ask(
    question: str,
    *,
    token_limit: int | None = None,
) -> tuple[str | None, types.GenerateContentResponseUsageMetadata | None]:
    client = genai.Client(api_key=get_api_key())
    raw_temperature = os.getenv("AIE_TEMPERATURE", "").strip()
    config = (
        types.GenerateContentConfig(temperature=float(raw_temperature))
        if raw_temperature
        else None
    )
    if token_limit is not None:
        ensure_within_token_limit(client, question, token_limit)
    response = client.models.generate_content(model=MODEL, contents=question, config=config)
    return response.text, response.usage_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    temperature = os.getenv("AIE_TEMPERATURE", "").strip()
    if temperature:
        print(f"[temperature] {temperature}")
    token_limit = parse_token_limit()
    if token_limit is not None:
        print(f"[token_limit] 入力上限 {token_limit}")
    print(f"[質問] {args.question}\n")
    try:
        answer, usage = ask(args.question, token_limit=token_limit)
    except TokenLimitExceeded as e:
        print(f"[tokens] 入力 {e.prompt_tokens} / 上限 {e.limit}")
        print("入力トークンが上限を超えたため、モデルは呼び出していません。")
        raise SystemExit(1)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)

    if usage is not None:
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
    print(answer)


if __name__ == "__main__":
    main()
