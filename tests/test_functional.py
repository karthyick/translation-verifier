from mtverify.gates import functional as f
from mtverify.gates import functional_gate
from mtverify.models import TestCase

SRC = "Hello {name}, your order #123 ships on Monday."


def _status(checks, name):
    return next(c for c in checks if c.name == name).status


def test_placeholders_many_styles():
    src = "Hi {{user}} ${a} %(n)s %s %1$s :tok {x}"
    assert f.check_placeholders(src, "Salut {{user}} ${a} %(n)s %s %1$s :tok {x}").status == "pass"
    bad = f.check_placeholders(src, "Salut {{user}} ${a} %(n)s %s %1$s :tok")
    assert bad.status == "fail" and "{x}" in bad.detail


def test_html_tags_normalised_and_counted():
    src = "Click <b>Pay</b> <br/> <a href='x'>here</a>"
    assert f.check_html_tags(src, "Klick <B>Zahlen</B> <br /> <a href='y'>hier</a>").status == "pass"
    bad = f.check_html_tags(src, "Klick Zahlen <br/> <a href='y'>hier</a>")
    assert bad.status == "fail" and "<b>" in bad.detail
    assert f.check_html_tags("no tags", "keine tags").status == "skip"


def test_numbers_accept_localised_digits_and_separators():
    assert f.check_numbers("Total 1,250.50 on 2026-10-08", "المجموع ١٬٢٥٠٫٥٠ في 2026-10-08").status == "pass"
    assert f.check_numbers("Total 1,250.50", "Gesamt 1.250,50").status == "pass"
    bad = f.check_numbers("Total 1,250.50", "Gesamt 1.520,50")
    assert bad.status == "fail"


def test_urls_and_emails_kept():
    src = "See https://x.io/a?b=1 or mail kr@x.io"
    assert f.check_urls_emails(src, "Siehe https://x.io/a?b=1 oder kr@x.io").status == "pass"
    assert f.check_urls_emails(src, "Siehe https://x.io/a oder kr@x.io").status == "fail"


def test_untranslated_detects_echo():
    assert f.check_untranslated(SRC, SRC, "en", "ta").status == "fail"
    assert f.check_untranslated("{name} #123", "{name} #123", "en", "ta").status == "skip"
    assert f.check_untranslated(SRC, "x", "en", "en").status == "skip"


def test_glossary_terms_required():
    assert f.check_glossary("Ouvrez le Dashboard", {"Dashboard": "Tableau de bord"}).status == "fail"
    assert f.check_glossary("Ouvrez le tableau de bord", {"Dashboard": "Tableau de bord"}).status == "pass"


def test_encoding_flags_replacement_char():
    assert f.check_encoding("ok �").status == "fail"
    assert f.check_encoding("ok").status == "pass"


def test_idempotent():
    assert f.check_idempotent("a", None).status == "skip"
    assert f.check_idempotent("a", "a").status == "pass"
    assert f.check_idempotent("a", "b").status == "fail"


def test_length_ratio_uses_language_bounds():
    assert f.check_length_ratio(SRC, "您好 {name}，订单 #123 周一发货。", "zh").status == "pass"
    assert f.check_length_ratio(SRC, "x", "de").status == "fail"


def test_functional_gate_good_and_bad():
    case = TestCase(id="t", source=SRC, target_lang="ta")
    good = functional_gate(case, "வணக்கம் {name}, உங்கள் ஆர்டர் #123 திங்கட்கிழமை அனுப்பப்படும்.")
    assert good.status == "pass"
    bad = functional_gate(case, "வணக்கம் பெயர், உங்கள் ஆர்டர் 123 செவ்வாய் அன்று அனுப்பப்படும்.")
    assert bad.status == "fail"
    assert _status(bad.checks, "placeholders") == "fail"
    assert _status(bad.checks, "language") == "pass"
