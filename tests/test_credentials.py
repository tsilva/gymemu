"""Local profile credentials must stay separate from explicit environment overrides."""

from types import SimpleNamespace

import pytest

from gymemu.credentials import r2_credentials


@pytest.fixture
def profile(tmp_path, monkeypatch):
    for name in ("ENDPOINT_URL", "ACCESS_KEY_ID", "SECRET_ACCESS_KEY"):
        monkeypatch.delenv(f"GYMEMU_MODELS_R2_{name}", raising=False)
    path = tmp_path / "r2.toml"
    path.write_text(
        'endpoint_url = "https://account.r2.cloudflarestorage.com"\n'
        '[keychain]\naccount = "gymemu"\naccess_key_id = "gymemu-access"\n'
        'secret_access_key = "gymemu-secret"\n'
    )
    path.chmod(0o600)
    monkeypatch.setenv("GYMEMU_R2_CONFIG", str(path))
    monkeypatch.setattr("gymemu.credentials.sys.platform", "darwin")
    return path


def test_profile_loads_named_keychain_credentials(profile, monkeypatch):
    calls = []

    def lookup(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=f"value-{args[3]}\n")

    monkeypatch.setattr("gymemu.credentials.subprocess.run", lookup)
    values = r2_credentials()
    assert values["ENDPOINT_URL"] == "https://account.r2.cloudflarestorage.com"
    assert values["ACCESS_KEY_ID"] == "value-gymemu-access"
    assert values["SECRET_ACCESS_KEY"] == "value-gymemu-secret"
    assert len(calls) == 2


def test_partial_environment_does_not_mix_with_profile(profile, monkeypatch):
    monkeypatch.setenv("GYMEMU_MODELS_R2_ACCESS_KEY_ID", "other-account-key")
    values = r2_credentials()
    assert values == {
        "ACCESS_KEY_ID": "other-account-key",
        "SECRET_ACCESS_KEY": "",
        "ENDPOINT_URL": "",
    }


def test_missing_keychain_entry_has_no_secret_in_error(profile, monkeypatch):
    monkeypatch.setattr(
        "gymemu.credentials.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="must-not-be-disclosed"),
    )
    with pytest.raises(ValueError, match="access_key_id is unavailable") as error:
        r2_credentials()
    assert "must-not-be-disclosed" not in str(error.value)
