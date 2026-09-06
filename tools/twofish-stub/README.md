# twofish stub

`pip install ./tools/twofish-stub` before `pip install "hopsworks[python]"` on Windows.

The Hopsworks SDK depends on `pyjks`, which depends on `twofish`, a C extension that
ships no Windows wheel for any Python version and needs the MSVC build tools to compile.
pyjks only imports twofish to decrypt BKS/UBER Java keystores, which Hopsworks uses for
in-cluster connections. An external connection - API key over HTTPS, which is the only
kind this project makes - never opens a keystore.

So this satisfies the dependency with a module that raises a clear ImportError if
anything ever actually calls it, instead of silently returning wrong bytes. Linux and
macOS do not need it; CI runs on Ubuntu and installs the real thing.
