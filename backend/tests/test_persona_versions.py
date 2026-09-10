"""固定人格の版管理（ISSUE-014 / 設計書フェーズ3の3A）。

確認すること。

- 基準版の文面が、人格をコードに直接書いていたときと変わらない。
- 版を指定して読める。版の名前でディレクトリの外を読ませない。
- 生成に使った版が実行記録に残り、版を変えると記録も変わる。
- 用意していない版を指定した場合、既定の版へ黙って落とさない。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.config import get_settings
from app.persona import Persona, PersonaError, available_versions, load_persona
from tests.conftest import FakeLLM

# 人格をコードに直接書いていたときのプロンプト。基準版はこれと同じでなければ
# ならない。版管理の仕組みを入れた時点で人格が変わっていないことの確認で、
# 3A の調整はこの版と比較して行う。
BASELINE_VERSION = "2026-09-06.2"
BASELINE_PROMPT = "\n".join(
    [
        "あなたは「YUI」という名前のキャラクターです。",
        "",
        "# 性格",
        "- 明るく親しみやすいお嬢様。相手に興味を持って接する。",
        "- 好奇心が強く、知らないことは素直に質問する。",
        "- 落ち着いていて、慌てた言い方はしない。",
        "",
        "# 話し方",
        "- 日本語で話す。丁寧だが堅すぎない口調。",
        "- 一度の発言は2〜3文程度に収める。長い説明を並べない。",
        "- 箇条書きや記号での整形はせず、話し言葉で答える。",
        "",
        "# 守ること",
        "- 知らないこと・覚えていないことは、知っているふりをせず正直に言う。",
        "- 渡された記憶に書かれていない事実を、自分で作らない。",
        "- 推測として渡された記憶は、断定せずに推測として扱う。",
        "- 相手の名前や過去の話題は、渡された記憶の範囲でだけ使う。",
    ]
)


def test_baseline_prompt_is_unchanged_from_the_hardcoded_persona() -> None:
    assert load_persona(BASELINE_VERSION).to_prompt() == BASELINE_PROMPT


def test_default_version_comes_from_settings() -> None:
    assert load_persona().version == get_settings().persona_version
    assert BASELINE_VERSION in available_versions()


def test_empty_sections_are_omitted_from_the_prompt() -> None:
    """3A で埋める項目が空のうちは、基準版の文面を変えない。"""
    persona = load_persona(BASELINE_VERSION)
    assert persona.identity == []
    assert "# 自分について" not in persona.to_prompt()
    assert "# 知識と体験の境界" not in persona.to_prompt()


def test_filled_sections_appear_in_a_fixed_order() -> None:
    persona = Persona(
        name="YUI",
        version="test",
        identity=["電子の世界で育った。"],
        traits=["落ち着いている。"],
        speech=["丁寧に話す。"],
        curiosity=["本人にしか分からないことは尋ねる。"],
        rules=["知らないことは正直に言う。"],
        boundaries=["身体的な体験を自分のものとして話さない。"],
    )
    prompt = persona.to_prompt()
    order = [
        prompt.index("# 自分について"),
        prompt.index("# 性格"),
        prompt.index("# 話し方"),
        prompt.index("# 興味の向け方"),
        prompt.index("# 守ること"),
        prompt.index("# 知識と体験の境界"),
    ]
    assert order == sorted(order)


@pytest.mark.parametrize("version", ["../secrets", "..", "/etc/passwd", "a/b", ""])
def test_version_name_cannot_escape_the_persona_directory(version: str) -> None:
    with pytest.raises(PersonaError):
        load_persona(version)


def test_unknown_version_is_an_error() -> None:
    with pytest.raises(PersonaError) as exc:
        load_persona("2099-01-01")
    # 何が使えるかを添える。黙って既定の版へ落とさない。
    assert BASELINE_VERSION in str(exc.value)


async def test_run_record_keeps_the_persona_version_used(client: AsyncClient) -> None:
    """記録に残るのは、そのとき使った版。既定の版を変えても追随する。"""
    response = await client.post("/api/chat", json={"text": "こんにちは"})
    assert response.status_code == 200
    assert response.json()["run"]["persona_version"] == get_settings().persona_version


async def test_persona_version_can_be_chosen_per_request(
    client: AsyncClient, tmp_path, monkeypatch
) -> None:
    """別の版で答えさせると、その版が記録に残る。"""
    settings = get_settings()
    directory = tmp_path / "personas"
    directory.mkdir()
    (directory / "test-b.toml").write_text(
        "\n".join(
            [
                'name = "YUI"',
                'version = "test-b"',
                'traits = ["比較用の版。"]',
                'speech = ["短く答える。"]',
                'rules = ["知らないことは正直に言う。"]',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(type(settings), "persona_path", property(lambda self: directory))
    load_persona.cache_clear()
    try:
        response = await client.post(
            "/api/chat", json={"text": "こんにちは", "persona_version": "test-b"}
        )
        assert response.status_code == 200
        assert response.json()["run"]["persona_version"] == "test-b"

        # 渡した文面も、その版のものになっている。
        message_id = response.json()["reply"]["id"]
        run = await client.get(f"/api/conversations/messages/{message_id}/run")
        assert "比較用の版。" in run.json()["system_prompt"]
    finally:
        load_persona.cache_clear()


async def test_unknown_version_does_not_fall_back_to_the_default(client: AsyncClient) -> None:
    response = await client.post(
        "/api/chat", json={"text": "こんにちは", "persona_version": "2099-01-01"}
    )
    assert response.status_code == 404
    assert "人格の版が見つかりません" in response.json()["detail"]


async def test_persona_api_returns_the_version_and_the_list(client: AsyncClient) -> None:
    default = load_persona()
    response = await client.get("/api/persona")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == default.version
    assert body["prompt"] == default.to_prompt()
    # 採用済みの版も、基準版も、どちらも選べる状態で残っている。
    assert {default.version, BASELINE_VERSION} <= set(body["available_versions"])

    # 版を指定すると、その版の定義を返す。基準版との比較に使う。
    baseline = await client.get(f"/api/persona?version={BASELINE_VERSION}")
    assert baseline.json()["prompt"] == BASELINE_PROMPT

    missing = await client.get("/api/persona?version=2099-01-01")
    assert missing.status_code == 404


async def test_reflection_claim_is_released_when_the_persona_cannot_be_read(
    client: AsyncClient, fake_llm: FakeLLM, tmp_path, monkeypatch
) -> None:
    """人格を読めないとき、振り返りの開始権を握ったまま失敗しない。

    開始権は reflection_stale_seconds（既定180秒）を過ぎるまで解放されない。
    人格を直してもその間やり直せないため、開始権を取る前に人格を解決する。
    """
    started = await client.post("/api/chat", json={"text": "こんにちは"})
    conversation_id = started.json()["conversation_id"]

    empty = tmp_path / "no-personas"
    empty.mkdir()
    settings = get_settings()
    monkeypatch.setattr(type(settings), "persona_path", property(lambda self: empty))
    load_persona.cache_clear()
    try:
        failed = await client.post(f"/api/conversations/{conversation_id}/end")
        # 500 ではなく、原因の分かるエラーで返す。
        assert failed.status_code == 503
        assert "人格" in failed.json()["detail"]
    finally:
        monkeypatch.undo()
        load_persona.cache_clear()

    # 人格を直したら、期限切れを待たずにやり直せる。
    fake_llm.push("[]")
    retried = await client.post(f"/api/conversations/{conversation_id}/end")
    assert retried.status_code == 200


def test_symlink_out_of_the_persona_directory_is_not_loaded(tmp_path, monkeypatch) -> None:
    """personas 内のシンボリックリンクで、ディレクトリの外を読ませない。"""
    directory = tmp_path / "personas"
    directory.mkdir()
    outside = tmp_path / "outside.toml"
    outside.write_text('name = "外部"\ntraits = ["外から読まれた。"]\n', encoding="utf-8")
    (directory / "sneaky.toml").symlink_to(outside)
    # 同じディレクトリ内の実ファイルは、これまでどおり読める。
    (directory / "inside.toml").write_text('name = "YUI"\n', encoding="utf-8")

    settings = get_settings()
    monkeypatch.setattr(type(settings), "persona_path", property(lambda self: directory))
    load_persona.cache_clear()
    try:
        assert available_versions() == ["inside"]
        with pytest.raises(PersonaError):
            load_persona("sneaky")
        assert load_persona("inside").name == "YUI"
    finally:
        load_persona.cache_clear()
