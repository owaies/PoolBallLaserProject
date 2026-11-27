from backend.app.core.config import Settings


def test_admin_credentials_are_not_hardcoded_by_default():
    settings = Settings(_env_file=None)

    assert settings.ADMIN_PASSWORD is None
    assert settings.ADMIN_JWT_SECRET is None
