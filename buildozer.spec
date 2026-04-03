[app]

# ── Basic info ────────────────────────────────────────────────────────────────

title           = NeuroGaming
package.name    = neurogaming
package.domain  = org.matpomgit

source.dir      = .
source.include_exts = py,kv,png,jpg,ttf,atlas
source.exclude_dirs = tests,.github,bin,.buildozer,__pycache__,.git

version         = 1.0.0

# ── Entry point ───────────────────────────────────────────────────────────────

# Kivy is the app framework; main.py is the entry point.
entrypoint = main.py

# ── Python requirements ───────────────────────────────────────────────────────

requirements = python3,kivy==2.3.0,numpy,bleak

# ── Orientation & display ─────────────────────────────────────────────────────

orientation  = portrait
fullscreen   = 0

# ── Android-specific settings ─────────────────────────────────────────────────

android.permissions = \
    BLUETOOTH, \
    BLUETOOTH_ADMIN, \
    BLUETOOTH_SCAN, \
    BLUETOOTH_CONNECT, \
    ACCESS_FINE_LOCATION, \
    ACCESS_COARSE_LOCATION

android.api       = 33
android.minapi    = 21
android.ndk       = 25b
android.sdk       = 33

android.arch = arm64-v8a

# Enable Bluetooth features
android.features = android.hardware.bluetooth_le

# ── iOS (placeholder – not built in CI) ──────────────────────────────────────

# ios.kivy_ios_url  = https://github.com/kivy/kivy-ios
# ios.kivy_ios_branch = master

# ── Buildozer / p4a options ───────────────────────────────────────────────────

[buildozer]

log_level = 2
warn_on_root = 1
