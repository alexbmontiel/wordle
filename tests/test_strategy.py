from wordle_helper.core.result import compute_result


class TestComputeResult:
    def test_exact_match(self):
        assert compute_result("CRANE", "CRANE") == "GGGGG"

    def test_no_match(self):
        assert compute_result("CRANE", "DUMPY") == "NNNNN"

    def test_mixed_result(self):
        # CRANE vs DRAIN: C=N, R=G, A=G, N=Y, E=N
        assert compute_result("CRANE", "DRAIN") == "NGGYN"

    def test_yellow_positions(self):
        # HELLO vs WORLD: H=N, E=N, L=N (consumed by green), L=G, O=Y
        assert compute_result("HELLO", "WORLD") == "NNNGY"

    def test_duplicate_in_guess_two_in_answer(self):
        # ALLOY vs LABEL: A=Y, L=Y, L=Y (LABEL has 2 L's), O=N, Y=N
        assert compute_result("ALLOY", "LABEL") == "YYYNN"

    def test_duplicate_in_answer(self):
        # SPEED vs CREEP: S=N, P=Y, E=G, E=G, D=N
        assert compute_result("SPEED", "CREEP") == "NYGGN"

    def test_green_takes_priority_over_yellow(self):
        # MAMMA vs LLAMA: M=N, A=Y, M=N (M at pos 3 is green in answer), M=G, A=G
        assert compute_result("MAMMA", "LLAMA") == "NYNGG"