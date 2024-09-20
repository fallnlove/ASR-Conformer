from src.metrics.utils import calc_cer, calc_wer


def test_cer():
    assert calc_cer("zvuuuuk", "zvuuuuk") == 0
    assert calc_cer("aboba", "oboba") == 1 / 5
    assert calc_cer("hse", "he") == 1 / 3
    assert calc_cer("computer science", "computer sciencee") == 1 / 16
    assert calc_cer("computer science", "") == 1
    assert calc_cer("com", "aaabbb") == 2


def test_wer():
    assert calc_wer("zvuuuuk zvuuk", "zvuuuuk zvuuk") == 0
    assert calc_wer("moscow hse", "hse") == 1 / 2
    assert calc_wer("hse", "wow broooo") == 2
    assert (
        calc_wer("hdi lab intern po xuyne ne pisat", "hdi lab intern po xuyne pisat")
        == 1 / 7
    )
    assert calc_wer("computer", "") == 1
