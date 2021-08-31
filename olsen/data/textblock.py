from typing import List


class TextBlock:
    def __init__(
            self, lines: List[str], begin: List[str], end: List[str], overlap: int
    ):
        self.lines = lines
        assert len(self.lines) >= overlap, f"Length is {len(self.lines)}"
        self.begin = ["<<<begin>>>"] * (overlap - len(begin))
        self.begin.extend(begin)
        assert len(self.begin) == overlap, f"Length is {len(self.begin)}"
        self.end = end
        self.end.extend(["<<<end>>>"] * (overlap - len(end)))
        assert len(self.end) == overlap, f"Length is {len(self.end)}"
