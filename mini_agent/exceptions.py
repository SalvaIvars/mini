class InterruptAgentFlow(Exception):
    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class LimitsExceeded(InterruptAgentFlow):
    pass


class FormatError(InterruptAgentFlow):
    pass
