import pytest

from mtverify.config import Settings
from mtverify.models import TestCase

SRC = "Hello {name}, your order #123 ships on Monday."


@pytest.fixture
def settings() -> Settings:
    return Settings(deepl_key="k:fx", google_key="g", azure_key="a", azure_region="eastus",
                    timeout_s=5, retries=2)


@pytest.fixture
def ta_case() -> TestCase:
    return TestCase(id="ta", source=SRC, target_lang="ta",
                    reference="வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கட்கிழமை அனுப்பப்படும்.")
