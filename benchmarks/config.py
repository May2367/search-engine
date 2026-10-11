from dataclasses import dataclass, field


CONTROLLED_TERMS = {
    "commonfrequency": (0.20, 1, 1),
    "mediumfrequency": (0.05, 1, 1),
    "rarefrequency": (0.01, 1, 1),
    "veryrarefrequency": (0.001, 1, 1),
    "repeatedterm": (0.01, 2, 3),
    "overlapalpha": (0.05, 1, 1),
    "overlapbeta": (0.05, 1, 1),
    "overlapgamma": (0.05, 1, 1),
}


@dataclass(frozen=True)
class TermConfig:
    df: float
    min_tf: int = 1
    max_tf: int = 1

    def __post_init__(self):
        if not 0 <= self.df <= 1:
            raise ValueError("df must be between 0 and 1")
        if self.min_tf < 1 or self.max_tf < self.min_tf:
            raise ValueError("Invalid term frequency range")


@dataclass(frozen=True)
class CorpusConfig:
    num_documents: int = 1_000
    seed: int = 42
    min_length: int = 4
    background_vocabulary_size: int = 2_992
    benchmark_terms: dict[str, TermConfig] = field(
        default_factory=lambda: {
            name: TermConfig(df, min_tf, max_tf)
            for name, (df, min_tf, max_tf) in CONTROLLED_TERMS.items()
        }
    )

    def __post_init__(self):
        if self.num_documents < 1:
            raise ValueError("num_documents must be positive")
        if self.min_length < 1:
            raise ValueError("min_length must be positive")
        if self.background_vocabulary_size != 2_992:
            raise ValueError("Expected 2,992 background vocabulary terms")
        if len(self.benchmark_terms) != 8:
            raise ValueError("Expected exactly 8 controlled terms")

        for term, settings in self.benchmark_terms.items():
            if not term.isalpha() or term.lower() != term:
                raise ValueError(
                    f"Controlled term must be lowercase alphabetic: {term}"
                )
            if not isinstance(settings, TermConfig):
                raise TypeError(f"Invalid configuration for {term}")


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_runs: int = 2
    measured_runs: int = 10

    def __post_init__(self):
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.measured_runs < 1:
            raise ValueError("measured_runs must be positive")

