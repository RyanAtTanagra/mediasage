"""Tests for API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.models import DefaultsConfig


@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Import here to avoid module-level import issues
    from backend.main import app
    return TestClient(app)


def create_mock_config(
    plex_url="http://test:32400",
    plex_token="token",
    music_library="Music",
    llm_provider="anthropic",
    llm_api_key="key",
    model_analysis="claude-sonnet-4-5",
    model_generation="claude-haiku-4-5",
    track_count=25,
    ollama_url="http://localhost:11434",
    custom_url="",
    custom_context_window=32768,
):
    """Create a properly structured mock config."""
    mock = MagicMock()
    mock.media_server = "plex"
    mock.jellyfin.url = ""
    mock.jellyfin.token = ""
    mock.jellyfin.music_library = "Music"
    mock.plex.url = plex_url
    mock.plex.token = plex_token
    mock.plex.music_library = music_library
    mock.llm.provider = llm_provider
    mock.llm.api_key = llm_api_key
    mock.llm.model_analysis = model_analysis
    mock.llm.model_generation = model_generation
    mock.llm.ollama_url = ollama_url
    mock.llm.custom_url = custom_url
    mock.llm.custom_context_window = custom_context_window
    mock.defaults = DefaultsConfig(track_count=track_count)
    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Should return health status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy"

    def test_health_check_shows_plex_status(self, client):
        """Should show Plex connection status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config()
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "plex_connected" in data
                assert data["plex_connected"] is True

    def test_health_check_shows_llm_status(self, client):
        """Should show LLM configuration status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_current_media_client") as mock_plex:
                mock_config.return_value = create_mock_config(llm_api_key="key")
                mock_plex.return_value = None  # No Plex client

                response = client.get("/api/health")

                assert response.status_code == 200
                data = response.json()
                assert "llm_configured" in data
                assert data["llm_configured"] is True


class TestConfigEndpoints:
    """Tests for configuration endpoints."""

    def test_get_config_returns_safe_values(self, client):
        """GET /api/config should return config without secrets."""
        with patch("backend.main.get_config") as mock_get_config:
            with patch("backend.main.get_plex_client") as mock_plex:
                mock_get_config.return_value = create_mock_config(
                    plex_url="http://test:32400",
                    plex_token="secret-token",
                    llm_provider="anthropic",
                    llm_api_key="secret-api-key",
                )
                mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                response = client.get("/api/config")

                assert response.status_code == 200
                data = response.json()

                # Should include URL but not token
                assert data["plex_url"] == "http://test:32400"
                assert "secret-token" not in str(data)

                # Should show provider but not API key
                assert data["llm_provider"] == "anthropic"
                assert "api_key" not in data
                assert "secret-api-key" not in str(data)

    def test_post_config_validates_plex_url(self, client):
        """POST /api/config should validate Plex URL format."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(plex_url="http://new-server:32400")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"plex_url": "http://new-server:32400"}
                    )

                    assert response.status_code == 200

    def test_post_config_updates_llm_provider(self, client):
        """POST /api/config should allow changing LLM provider."""
        with patch("backend.main.update_config_values") as mock_update:
            with patch("backend.main.get_plex_client") as mock_plex:
                with patch("backend.main.init_plex_client"):
                    mock_config = create_mock_config(llm_provider="openai")
                    mock_update.return_value = mock_config
                    mock_plex.return_value = MagicMock(is_connected=MagicMock(return_value=True))

                    response = client.post(
                        "/api/config",
                        json={"llm_provider": "openai"}
                    )

                    assert response.status_code == 200


class TestIndexPage:
    """Tests for index page serving."""

    def test_index_returns_response(self, client):
        """Should return some response for root path."""
        response = client.get("/")
        # Either returns HTML or JSON message
        assert response.status_code == 200


class TestOllamaEndpoints:
    """Tests for Ollama API endpoints."""

    def test_ollama_status_connected(self, client):
        """GET /api/ollama/status should return connected status."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=3,
                    error=None,
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is True
                assert data["model_count"] == 3

    def test_ollama_status_not_connected(self, client):
        """GET /api/ollama/status should return error when not connected."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=False,
                    model_count=0,
                    error="Connection refused",
                )

                response = client.get("/api/ollama/status")

                assert response.status_code == 200
                data = response.json()
                assert data["connected"] is False
                assert data["error"] == "Connection refused"

    def test_ollama_models_list(self, client):
        """GET /api/ollama/models should return list of models."""
        from backend.models import OllamaModel, OllamaModelsResponse

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.list_ollama_models") as mock_models:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_models.return_value = OllamaModelsResponse(
                    models=[
                        OllamaModel(name="llama3:8b", size=4661224676, modified_at="2024-01-15T00:00:00Z"),
                        OllamaModel(name="mistral:latest", size=3825819904, modified_at="2024-01-14T00:00:00Z"),
                    ],
                    error=None,
                )

                response = client.get("/api/ollama/models")

                assert response.status_code == 200
                data = response.json()
                assert len(data["models"]) == 2
                assert data["models"][0]["name"] == "llama3:8b"

    def test_ollama_model_info(self, client):
        """GET /api/ollama/model-info should return model details."""
        from backend.models import OllamaModelInfo

        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = OllamaModelInfo(
                    name="llama3:8b",
                    context_window=8192,
                    parameter_size="8B",
                )

                response = client.get("/api/ollama/model-info?model=llama3:8b")

                assert response.status_code == 200
                data = response.json()
                assert data["name"] == "llama3:8b"
                assert data["context_window"] == 8192

    def test_ollama_model_info_not_found(self, client):
        """GET /api/ollama/model-info should return 404 for unknown model."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_model_info") as mock_info:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_info.return_value = None  # Model not found

                response = client.get("/api/ollama/model-info?model=nonexistent")

                assert response.status_code == 404

    def test_ollama_status_with_custom_url(self, client):
        """GET /api/ollama/status should accept custom URL parameter."""
        with patch("backend.main.get_config") as mock_config:
            with patch("backend.main.get_ollama_status") as mock_status:
                mock_config.return_value = create_mock_config(
                    llm_provider="ollama",
                    ollama_url="http://localhost:11434",
                )
                mock_status.return_value = MagicMock(
                    connected=True,
                    model_count=1,
                    error=None,
                )

                response = client.get("/api/ollama/status?url=http://custom-host:11434")

                assert response.status_code == 200
                # Verify the custom URL was passed
                mock_status.assert_called_once_with("http://custom-host:11434")


class TestCloudModelsEndpoint:
    """Tests for GET /api/models."""

    def test_lists_models_per_provider(self, client):
        """Should return each cloud provider's models with its defaults."""
        response = client.get("/api/models")

        assert response.status_code == 200
        providers = response.json()["providers"]
        assert set(providers) == {"anthropic", "openai", "gemini"}
        for catalog in providers.values():
            ids = [m["id"] for m in catalog["models"]]
            assert catalog["default_analysis"] in ids
            assert catalog["default_generation"] in ids

    def test_model_entries_include_limits_and_prices(self, client):
        """Entries should carry context, track capacity and prices for the UI."""
        response = client.get("/api/models")

        gemini = response.json()["providers"]["gemini"]["models"]
        flash_lite = next(m for m in gemini if m["id"] == "gemini-3.5-flash-lite")
        assert flash_lite["context_window"] == 1_000_000
        assert flash_lite["max_tracks"] > 20_000
        assert flash_lite["cost_per_million_input"] == 0.30
        assert flash_lite["legacy"] is False


class TestConfigLongContextPricing:
    """Tests for long-context pricing in GET /api/config."""

    def test_included_for_models_with_long_pricing(self, client):
        mock_config = create_mock_config(
            llm_provider="openai", model_analysis="gpt-6.1-sol", model_generation="gpt-6-luna"
        )
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_plex_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["long_context_threshold"] == 272_000
        assert data["long_context_cost_per_million_input"] == 0.20

    def test_absent_for_flat_priced_models(self, client):
        with patch("backend.main.get_config", return_value=create_mock_config()), \
             patch("backend.main.get_plex_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["long_context_threshold"] is None


class TestLibraryStatsJellyfin:
    """Jellyfin stats come from the cache once synced (scanning Jellyfin takes about a minute)."""

    def test_uses_cache_when_synced(self, client):
        mock_config = create_mock_config()
        mock_config.media_server = "jellyfin"
        jellyfin = MagicMock(is_connected=MagicMock(return_value=True))
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_current_media_client", return_value=jellyfin), \
             patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.get_sync_state", return_value={"track_count": 42}), \
             patch("backend.main.library_cache.get_cached_genre_decade_stats",
                   return_value={"genres": [{"name": "Jazz", "count": 42}], "decades": []}):
            data = client.get("/api/library/stats").json()

        assert data["total_tracks"] == 42
        assert data["genres"] == [{"name": "Jazz", "count": 42}]
        jellyfin.get_library_stats.assert_not_called()


class TestMediaServerSwitch:
    """Switching between Plex and Jellyfin replaces the cached library."""

    def _post_config(self, client, previous, new, updates):
        before = create_mock_config()
        before.media_server = previous
        after = create_mock_config()
        after.media_server = new
        with patch("backend.main.get_config", return_value=before), \
             patch("backend.main.update_config_values", return_value=after), \
             patch("backend.main.init_plex_client"), \
             patch("backend.main.init_jellyfin_client"), \
             patch("backend.main.get_current_media_client", return_value=None), \
             patch("backend.main._resync_for_new_media_server") as resync:
            response = client.post("/api/config", json=updates)
        assert response.status_code == 200
        return resync

    def test_switching_server_resyncs(self, client):
        resync = self._post_config(client, "jellyfin", "plex", {"media_server": "plex"})
        resync.assert_called_once()

    def test_saving_same_server_does_not_resync(self, client):
        resync = self._post_config(client, "plex", "plex", {"media_server": "plex", "music_library": "Music"})
        resync.assert_not_called()

    def test_plex_setup_selects_plex_and_resyncs_from_jellyfin(self, client):
        temp_client = MagicMock(is_connected=MagicMock(return_value=True))
        temp_client.get_music_libraries.return_value = ["Music"]
        temp_client._server.friendlyName = "My Plex Server"
        previous = create_mock_config()
        previous.media_server = "jellyfin"
        with patch("backend.main.PlexClientInstance", return_value=temp_client), \
             patch("backend.main.get_config", return_value=previous), \
             patch("backend.main.update_config_values") as update, \
             patch("backend.main.init_plex_client"), \
             patch("backend.main._resync_for_new_media_server") as resync:
            response = client.post("/api/setup/validate-plex", json={
                "plex_url": "http://plex:32400", "plex_token": "abc", "music_library": "Music",
            })

        assert response.json()["success"] is True
        assert update.call_args.args[0]["media_server"] == "plex"
        resync.assert_called_once()

    def test_resync_invalidates_now_and_replaces_in_background(self):
        from backend.main import _resync_for_new_media_server

        with patch("backend.main.library_cache") as cache, \
             patch("backend.main.asyncio.to_thread", new=MagicMock()) as to_thread, \
             patch("backend.main.asyncio.create_task"):
            _resync_for_new_media_server()

        cache.invalidate_cache.assert_called_once()
        cache.clear_cache.assert_not_called()  # a running sync holds the DB; never block the request
        to_thread.assert_called_once()
        assert to_thread.call_args.args[0] is cache.replace_library
        assert to_thread.call_args.args[2] == cache.invalidate_cache.return_value


class TestArtistEndpoints:
    """Artist autocomplete and artist filters in the preview (#9)."""

    def test_artists_empty_without_cache(self, client):
        with patch("backend.main.library_cache.has_cached_tracks", return_value=False):
            assert client.get("/api/library/artists?q=radio").json() == []

    def test_artists_search(self, client):
        with patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.search_artists",
                   return_value=[{"name": "Radiohead", "count": 2}]) as search:
            data = client.get("/api/library/artists?q=%20radio%20").json()

        assert data == [{"name": "Radiohead", "count": 2}]
        search.assert_called_once_with("radio")

    def test_preview_passes_artist_filters(self, client):
        with patch("backend.main.get_config", return_value=create_mock_config()), \
             patch("backend.main.library_cache.has_cached_tracks", return_value=True), \
             patch("backend.main.library_cache.count_tracks_by_filters", return_value=12) as count:
            data = client.post("/api/filter/preview", json={
                "genres": [], "decades": [], "artists": ["Radiohead"], "exclude_artists": ["Oasis"],
            }).json()

        assert data["matching_tracks"] == 12
        assert count.call_args.kwargs["artists"] == ["Radiohead"]
        assert count.call_args.kwargs["exclude_artists"] == ["Oasis"]


class TestLibraryFreshness:
    """The library cache re-syncs once it's older than the auto-refresh setting (default 24 hours)."""

    def _run(self, *, stale=True, connected=True, syncing=False, hours=24.0):
        from backend.main import _sync_if_stale

        config = create_mock_config()
        config.library_sync_hours = hours
        media_client = MagicMock(is_connected=MagicMock(return_value=connected))
        with patch("backend.main.get_config", return_value=config), \
             patch("backend.main.library_cache") as cache, \
             patch("backend.main.get_current_media_client", return_value=media_client), \
             patch("backend.main.asyncio.to_thread", new=MagicMock()) as to_thread, \
             patch("backend.main.asyncio.create_task"):
            cache.get_sync_progress.return_value = {"is_syncing": syncing}
            cache.is_cache_stale.return_value = stale
            _sync_if_stale()
        return cache, to_thread

    def test_syncs_stale_cache(self):
        cache, to_thread = self._run()
        cache.is_cache_stale.assert_called_once_with(24.0)
        assert to_thread.call_args.args[0] is cache.sync_library

    def test_leaves_fresh_cache(self):
        _, to_thread = self._run(stale=False)
        to_thread.assert_not_called()

    def test_skips_when_disconnected_or_already_syncing(self):
        assert not self._run(connected=False)[1].called
        assert not self._run(syncing=True)[1].called

    def test_custom_interval_and_off(self):
        cache, _ = self._run(hours=168.0)
        cache.is_cache_stale.assert_called_once_with(168.0)
        cache, to_thread = self._run(hours=0)
        cache.is_cache_stale.assert_not_called()
        to_thread.assert_not_called()


class TestApiKeyProviders:
    def test_lists_providers_with_a_key(self, client, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CUSTOM_LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        mock_config = create_mock_config(llm_provider="gemini")
        mock_config.llm.api_keys = {"gemini": "saved"}
        with patch("backend.main.get_config", return_value=mock_config), \
             patch("backend.main.get_current_media_client", return_value=None):
            data = client.get("/api/config").json()

        assert data["api_key_providers"] == ["openai", "gemini"]
