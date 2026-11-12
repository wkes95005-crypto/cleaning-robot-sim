"""Simulated wireless link between the robots and the console: every message arrives `latency` ticks late."""
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str
    to: str
    topic: str
    payload: dict = field(default_factory=dict)
    sent: int = 0


class Bus:
    def __init__(self, latency=1):
        self.latency = latency
        self.tick = 0
        self.sent = 0
        self._queues = defaultdict(list)

    def send(self, sender, to, topic, payload=None):
        msg = Message(sender, to, topic, payload or {}, self.tick)
        self._queues[to].append((self.tick + self.latency, msg))
        self.sent += 1

    def receive(self, to):
        queue = self._queues[to]
        ready = [m for t, m in queue if t <= self.tick]
        self._queues[to] = [(t, m) for t, m in queue if t > self.tick]
        return ready
