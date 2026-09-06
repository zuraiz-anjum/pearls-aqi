"""Placeholder for the twofish C extension.

pyjks declares twofish as a hard dependency but only imports it to decrypt BKS/UBER
keystores. Hopsworks external connections authenticate with an API key over HTTPS and
never open a keystore, so on Windows - where twofish ships no wheel and needs MSVC to
build - this stub lets the SDK install. Anything that genuinely needs the cipher gets
a clear error instead of a silent wrong answer.
"""


class Twofish:  # noqa: N801 - matches the real module's class name
    def __init__(self, *_, **__):
        raise ImportError(
            "twofish is a stub in this environment; install the real package "
            "(needs MSVC build tools on Windows) if you actually need the cipher"
        )
