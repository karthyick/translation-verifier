import pytest

from mtverify import languages


def test_registry_has_at_least_20_languages():
    assert len(languages.REGISTRY) >= 20
    for lang in languages.REGISTRY.values():
        assert lang.ranges, lang.code
        assert lang.length_ratio[0] < lang.length_ratio[1]


@pytest.mark.parametrize(
    "code,text",
    [
        ("ta", "வணக்கம் {name}, உங்கள் ஆர்டர் #123"),
        ("kn", "ನಮಸ್ಕಾರ {name}, ನಿಮ್ಮ ಆರ್ಡರ್ #123"),  # lingua cannot do kn: script path
        ("ml", "ഹലോ {name}, നിങ്ങളുടെ ഓർഡർ #123"),
        ("hi", "नमस्ते {name}, आपका ऑर्डर #123 सोमवार को भेजा जाएगा।"),
        ("ar", "مرحباً {name}، سيتم شحن طلبك #123 يوم الاثنين."),
        ("zh", "您好 {name}，您的订单 #123 将于周一发货。"),
        ("ja", "こんにちは {name}、ご注文 #123 は月曜日に発送されます。"),
        ("ko", "안녕하세요 {name}, 주문 #123은 월요일에 발송됩니다."),
        ("de", "Hallo {name}, Ihre Bestellung #123 wird am Montag versendet."),
        ("ru", "Здравствуйте, {name}, ваш заказ #123 будет отправлен."),
    ],
)
def test_detect_accepts_correct_language(code, text):
    d = languages.detect(text, code)
    assert d.ok, d


def test_detect_rejects_wrong_script():
    d = languages.detect("Hello {name}, your order #123 ships on Monday.", "hi")
    assert not d.ok and d.method == "script"


def test_detect_rejects_same_script_neighbour():
    hindi = "नमस्ते, आपका ऑर्डर सोमवार को भेजा जाएगा। यह एक लंबा हिंदी वाक्य है।"
    d = languages.detect(hindi, "mr")
    assert not d.ok and d.detected == "hi"


def test_detect_passes_when_only_placeholders():
    assert languages.detect("{name} #123", "ta").ok


def test_unknown_language_lists_options():
    with pytest.raises(languages.UnknownLanguage, match="not in registry"):
        languages.get("xx")
