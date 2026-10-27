"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
ローカルではリポジトリ直下の .env が、未設定の変数だけを補う。
すでに入っている環境変数（Codespaces secret やシェルの export）は上書きしない。
"""

import os
from pathlib import Path


def _parse_dotenv(text: str) -> dict[str, str]:
    """KEY=VALUE の行だけを取り出す。空の値、コメント、代入でない行は捨てる。"""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and value:
            values[key] = value
    return values


def _load_dotenv(path: Path | None = None) -> None:
    """リポジトリ直下の .env を読む。既に環境変数にある名前は変えない。"""
    env_path = Path(__file__).resolve().parents[1] / ".env" if path is None else path
    if not env_path.is_file():
        return
    for key, value in _parse_dotenv(env_path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)


_load_dotenv()

# 既定モデル。環境変数 AIE_MODEL を設定すると切り替えられる。
#   例: export AIE_MODEL=gemini-3.5-flash
#   ローカルでは .env に AIE_MODEL=gemini-3.5-flash と書いても同じ。
DEFAULT_MODEL = "gemini-3.5-flash-lite"
MODEL = os.getenv("AIE_MODEL", DEFAULT_MODEL)


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        return "レート制限に達しました。1分ほど待ってから再実行してください。"
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_api_key() -> str:
    """環境変数 GEMINI_API_KEY を返す。未設定なら手順を表示して終了する。"""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(
            "環境変数 GEMINI_API_KEY が見つかりません。\n"
            "1. GitHub の Settings → Codespaces → Secrets に GEMINI_API_KEY を登録し、\n"
            "   使う repository に自分の fork を選んでください。\n"
            "2. Codespaces を起動したあとに登録した場合は、Codespaces を再起動してください。\n"
            "   （左下の「Codespaces」→「Stop Current Codespace」→ 再度開く）\n"
            "3. 自分の PC で動かす場合は、.env.example を .env にコピーし、\n"
            "   GEMINI_API_KEY= にキーを書いてください。.env はコミットされません。"
        )
    return key
