"""
Audio utilities package for Fulloch voice assistant.

Voice I/O is now exclusively through the WebSocket satellite
(`/ws/satellite`); the local-mic + local-speaker paths are gone. This
package is kept as a placeholder for any future audio-side helpers
that don't belong in `core/` (e.g. audio-format conversions, codec
wrappers).
"""
