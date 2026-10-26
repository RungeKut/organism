from __future__ import annotations

from dataclasses import dataclass

KINDS = ("вопрос", "ответ", "факт", "просьба", "отказ")
KEYS = ("кому", "тип", "уверенность", "текст")


@dataclass
class Envelope:
    to: str
    kind: str
    text: str
    confidence: float = 1.0
    sender: str = ""

    def __post_init__(self) -> None:
        self.to = self.to.strip().lower()
        self.kind = self.kind.strip().lower()
        if self.kind not in KINDS:
            raise ValueError(f"неизвестный тип «{self.kind}»")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> dict:
        return {"от": self.sender, "кому": self.to, "тип": self.kind,
                "уверенность": round(self.confidence, 3), "текст": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> Envelope:
        return cls(d["кому"], d["тип"], d.get("текст", ""), d.get("уверенность", 1.0), d.get("от", ""))

    def render(self) -> str:
        return f"кому: {self.to}\nтип: {self.kind}\nуверенность: {self.confidence:.2f}\nтекст: {self.text}"

    @classmethod
    def parse(cls, raw: str) -> Envelope | None:
        head: dict[str, str] = {}
        body: list[str] | None = None
        for line in raw.strip().splitlines():
            if body is not None:
                body.append(line)
                continue
            key, sep, value = line.partition(":")
            key = key.strip().lower()
            if not sep or key not in KEYS:
                continue
            if key == "текст":
                body = [value.strip()]
            else:
                head[key] = value.strip()
        if body is None or "кому" not in head or "тип" not in head:
            return None
        try:
            confidence = float(head.get("уверенность", "1").replace(",", "."))
        except ValueError:
            confidence = 0.5
        try:
            return cls(head["кому"], head["тип"], "\n".join(body).strip(), confidence)
        except ValueError:
            return None
