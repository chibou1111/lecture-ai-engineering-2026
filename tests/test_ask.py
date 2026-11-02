import weakref
from types import SimpleNamespace

from app.ask import ask, ask_interactively


class FakeChat:
    def __init__(self):
        self.questions = []

    def send_message(self, question):
        self.questions.append(question)
        return SimpleNamespace(
            text=f"回答: {question}",
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
        )


def test_interactive_questions_use_the_same_chat(monkeypatch, capsys):
    questions = iter(["最初の質問", "続けて質問", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    chat = FakeChat()

    ask_interactively(chat)

    assert chat.questions == ["最初の質問", "続けて質問"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 続けて質問" in output
    assert output.count("[トークン数] 入力: 1, 出力: 2") == 2


def test_single_question_displays_token_counts_and_keeps_client_alive(monkeypatch, capsys):
    client_refs = []

    class Client:
        def __init__(self, *, api_key: str):
            assert api_key == "test"
            client_refs.append(weakref.ref(self))
            self.models = Models(client_refs[-1])

    class Models:
        def __init__(self, client_ref):
            self.client_ref = client_ref

        def generate_content(self, *, model: str, contents: str):
            assert model
            assert contents == "質問"
            assert self.client_ref() is not None
            return SimpleNamespace(
                text="回答: 質問",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=3,
                    candidates_token_count=5,
                ),
            )

    monkeypatch.setattr("app.ask.genai.Client", Client)
    monkeypatch.setattr("app.ask.get_api_key", lambda: "test")

    assert ask("質問") == "回答: 質問"
    assert "[トークン数] 入力: 3, 出力: 5" in capsys.readouterr().out
