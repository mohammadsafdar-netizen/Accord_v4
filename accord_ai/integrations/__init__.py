"""Third-party integrations (opt-in).

Everything in this package is gated by a *_enabled setting and falls back
to a null behavior when disabled, so offline dev and tests that don't
mock the integration still pass.
"""
