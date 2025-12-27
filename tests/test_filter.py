from wordle_helper.filter import Constraints, filter_word_list, ismatch, parse_response


class TestParseResponse:
    def test_all_greens(self):
        constraints = parse_response("CRANE", "GGGGG")
        assert constraints.exact == ((0, "C"), (1, "R"), (2, "A"), (3, "N"), (4, "E"))
        assert constraints.present == frozenset()
        assert constraints.absent_at == ()
        assert constraints.absent == frozenset()

    def test_all_greys(self):
        constraints = parse_response("CRANE", "NNNNN")
        assert constraints.exact == ()
        assert constraints.present == frozenset()
        assert constraints.absent_at == ()
        assert constraints.absent == frozenset({"C", "R", "A", "N", "E"})

    def test_all_yellows(self):
        constraints = parse_response("CRANE", "YYYYY")
        assert constraints.exact == ()
        assert constraints.present == frozenset({"C", "R", "A", "N", "E"})
        assert constraints.absent_at == ((0, "C"), (1, "R"), (2, "A"), (3, "N"), (4, "E"))
        assert constraints.absent == frozenset()

    def test_mixed_response(self):
        # guess CRANE, answer DRAIN: C=N, R=G, A=G, N=Y, E=N
        constraints = parse_response("CRANE", "NGGYN")
        assert constraints.exact == ((1, "R"), (2, "A"))
        assert constraints.present == frozenset({"N"})
        assert constraints.absent_at == ((3, "N"),)
        assert constraints.absent == frozenset({"C", "E"})

    def test_duplicate_letter_grey_then_yellow(self):
        # guess ALLOY, answer has one L at different position
        # A=N, L=Y, L=N, O=N, Y=N
        # The second L is grey but L should NOT be in absent (it's confirmed by first L)
        constraints = parse_response("ALLOY", "NYNNN")
        assert "L" in constraints.present
        assert "L" not in constraints.absent

    def test_duplicate_letter_yellow_then_grey(self):
        # Similar but yellow comes first in guess
        # guess LEVEL, answer LEVER: L=G, E=G, V=G, E=G, L=N
        # Last L is grey but L is confirmed by first position
        constraints = parse_response("LEVEL", "GGGGN")
        assert (0, "L") in constraints.exact
        assert "L" not in constraints.absent

    def test_duplicate_letter_grey_before_green(self):
        # guess LEMON, answer MELON: L=Y, E=G, M=Y, O=G, N=G
        constraints = parse_response("LEMON", "YGYGG")
        assert constraints.exact == ((1, "E"), (3, "O"), (4, "N"))
        assert constraints.present == frozenset({"L", "M"})
        assert "L" not in constraints.absent
        assert "M" not in constraints.absent


class TestIsmatch:
    def test_matches_exact_positions(self):
        constraints = Constraints(
            exact=((0, "D"), (1, "R")),
            present=frozenset(),
            absent_at=(),
            absent=frozenset(),
        )
        assert ismatch("DRAIN", frozenset("DRAIN"), constraints) is True
        assert ismatch("BRAIN", frozenset("BRAIN"), constraints) is False
        assert ismatch("DROLL", frozenset("DROLL"), constraints) is True

    def test_rejects_absent_letters(self):
        constraints = Constraints(
            exact=(),
            present=frozenset(),
            absent_at=(),
            absent=frozenset({"X", "Z"}),
        )
        assert ismatch("CRANE", frozenset("CRANE"), constraints) is True
        assert ismatch("AZURE", frozenset("AZURE"), constraints) is False
        assert ismatch("PROXY", frozenset("PROXY"), constraints) is False

    def test_requires_present_letters(self):
        constraints = Constraints(
            exact=(),
            present=frozenset({"A", "E"}),
            absent_at=(),
            absent=frozenset(),
        )
        assert ismatch("CRANE", frozenset("CRANE"), constraints) is True
        assert ismatch("BRAIN", frozenset("BRAIN"), constraints) is False  # no E
        assert ismatch("EAGLE", frozenset("EAGLE"), constraints) is True

    def test_rejects_absent_at_positions(self):
        constraints = Constraints(
            exact=(),
            present=frozenset(),
            absent_at=((0, "C"), (2, "A")),
            absent=frozenset(),
        )
        assert ismatch("CRANE", frozenset("CRANE"), constraints) is False  # C at 0
        assert ismatch("BRAIN", frozenset("BRAIN"), constraints) is False  # A at 2
        assert ismatch("DRINK", frozenset("DRINK"), constraints) is True

    def test_combined_constraints(self):
        # Simulates guess=CRANE result=NGGYN against answer DRAIN
        constraints = Constraints(
            exact=((1, "R"), (2, "A")),
            present=frozenset({"N"}),
            absent_at=((3, "N"),),
            absent=frozenset({"C", "E"}),
        )
        assert ismatch("DRAIN", frozenset("DRAIN"), constraints) is True
        assert ismatch("BRAID", frozenset("BRAID"), constraints) is False  # no N
        assert ismatch("CRANE", frozenset("CRANE"), constraints) is False  # has C
        assert ismatch("GROAN", frozenset("GROAN"), constraints) is False  # N at position 3 (absent_at)
        assert ismatch("TRAIN", frozenset("TRAIN"), constraints) is True


class TestFilterWordList:
    def test_filters_correctly(self):
        word_list: dict[str, tuple[float, frozenset[str]]] = {
            "DRAIN": (1.0, frozenset("DRAIN")),
            "CRANE": (0.9, frozenset("CRANE")),
            "BRAID": (0.8, frozenset("BRAID")),
            "TRAIN": (0.7, frozenset("TRAIN")),
            "GROAN": (0.6, frozenset("GROAN")),
        }
        # CRANE guessed, DRAIN is answer: NGGYN
        result = filter_word_list("CRANE", "NGGYN", word_list)

        assert "DRAIN" in result
        assert "TRAIN" in result
        assert "CRANE" not in result  # has C
        assert "BRAID" not in result  # no N in word
        assert "GROAN" not in result  # N at position 3

    def test_preserves_order(self):
        word_list: dict[str, tuple[float, frozenset[str]]] = {
            "AAAAA": (1.0, frozenset("AAAAA")),
            "BBBBB": (0.9, frozenset("BBBBB")),
            "CCCCC": (0.8, frozenset("CCCCC")),
        }
        result = filter_word_list("XXXXX", "NNNNN", word_list)
        assert list(result.keys()) == ["AAAAA", "BBBBB", "CCCCC"]

    def test_empty_result(self):
        word_list: dict[str, tuple[float, frozenset[str]]] = {
            "CRANE": (1.0, frozenset("CRANE")),
        }
        result = filter_word_list("CRANE", "GGGGG", word_list)
        # Only CRANE would match, and it does
        assert "CRANE" in result

    def test_returns_empty_when_no_matches(self):
        word_list: dict[str, tuple[float, frozenset[str]]] = {
            "CRANE": (1.0, frozenset("CRANE")),
        }
        # Require Z which CRANE doesn't have
        result = filter_word_list("ZEBRA", "GNNNN", word_list)
        assert len(result) == 0
