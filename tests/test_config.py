import pytest

from hay import config


def test_missing_vars_are_listed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env.local")
    for name in config.REQUIRED_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")

    with pytest.raises(config.ConfigError) as exc:
        config.load_config()

    message = str(exc.value)
    assert "OPENAI_API_KEY" in message and "PINECONE_INDEX" in message
    assert "TELEGRAM_BOT_TOKEN" not in message
