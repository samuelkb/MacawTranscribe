import logging


_STANDARD_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__.keys()) | {
    "message",
    "asctime",
}


class ExtraFieldsFormatter(logging.Formatter):
    """
    Append any non-standard LogRecord attributes to the rendered message.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_FIELDS
        }
        if not extras:
            return formatted

        extras_text = " ".join(
            f"{key}={value!r}" for key, value in sorted(extras.items())
        )
        return f"{formatted} {extras_text}"
