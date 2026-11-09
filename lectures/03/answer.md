# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
- `--trace` で見えたツール呼び出し（ツール名と引数）：
    [tool call]   search_glossary({'query': 'LoRA'})
    [tool call]   search_glossary({'query': 'QLoRA'})
- 1つの質問での LLM 呼び出し回数（`requests`）：2
- 追加したツールと、docstring に書いた説明： 
    - `make_quiz`：指定した回（1始まりの整数）の用語から1つを選び、「この用語の意味は？」という3択問題を一つ返す。
    - docstring にはいつ、何のために使うツールかを書いた。

## 宿題

- 取り組んだレベル：基礎
- やったこと：search_glossaryのDocstringを変更し、語尾を変えた。make_quiz を追加した。`--trace` で見えたツール呼び出しを確認した。
- 結果：語尾が「にゃん」になった。クイズも作成できた。
- 気づき・疑問：複数回ループが回るので、その分トークン消費が行われることに気がついた。
