
"""Pytest tests for the Korp /corpus_config endpoint."""


class TestCorpusConfig:
    """Tests for /corpus_config."""

    def test_corpus_config(self, get_json, corpus_configs):
        """Test that a corpus configuration can be retrieved."""
        data = get_json(
            "/corpus_config",
            params={
                "mode": "default",
                "cache": "false",
            },
        )
        # TODO: Add more assertions
        assert data["label"]
        assert data["corpora"]
        assert data["modes"]
        corpus_config = data["corpora"]["testcorpus"]
        assert corpus_config
        assert corpus_config["pos_attributes"]
        assert corpus_config["struct_attributes"]
