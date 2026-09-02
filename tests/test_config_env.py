from tlh.config import update_env_file


def test_update_env_preserves_comments_and_adds_keys(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# secrets\nANTHROPIC_API_KEY=\nTLH_AI_MODEL=claude-opus-5\n# tail comment\n", encoding="utf-8")
    update_env_file({"ANTHROPIC_API_KEY": "sk-test", "TLH_AI_EFFORT": "medium"}, p)
    txt = p.read_text(encoding="utf-8")
    assert "# secrets" in txt and "# tail comment" in txt
    assert "ANTHROPIC_API_KEY=sk-test" in txt and "TLH_AI_EFFORT=medium" in txt
    assert txt.count("ANTHROPIC_API_KEY=") == 1 and "TLH_AI_MODEL=claude-opus-5" in txt


def test_update_env_creates_file(tmp_path):
    p = tmp_path / "new.env"
    update_env_file({"A": "1"}, p)
    assert p.read_text(encoding="utf-8") == "A=1\n"
