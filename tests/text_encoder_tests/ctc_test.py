from src.text_encoder import CTCTextEncoder


def test_ctc_decode():
    encoder = CTCTextEncoder()

    assert (
        encoder.ctc_decode(
            [
                encoder.char2ind[v]
                for v in [
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    encoder.EMPTY_TOK,
                    encoder.EMPTY_TOK,
                    "o",
                    "b",
                    "u",
                    "u",
                    encoder.EMPTY_TOK,
                    "s",
                    " ",
                    " ",
                    " ",
                    encoder.EMPTY_TOK,
                    encoder.EMPTY_TOK,
                    "z",
                    "v",
                    encoder.EMPTY_TOK,
                    "u",
                    "u",
                    encoder.EMPTY_TOK,
                    "u",
                    encoder.EMPTY_TOK,
                    "u",
                    "u",
                    "u",
                    "k",
                ]
            ]
        )
        == "abobus zvuuuk"
    )
