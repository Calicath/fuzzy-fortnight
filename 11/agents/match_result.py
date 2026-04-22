"""
MatchResult - 匹配结果数据结构
"""


class MatchResult:
    def __init__(self, algorithm, found, confidence, location=None, size=None):
        self.algorithm = algorithm
        self.found = found
        self.confidence = confidence
        self.location = location
        self.size = size
        self.elapsed_ms = 0
        self.params = {}
