class InterruptFlow(Exception):
    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class LimitsExceeded(InterruptFlow):
    pass


class FormatError(InterruptFlow):
    pass
