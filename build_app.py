#!/usr/bin/env python3
"""
Build script for creating standalone MusicPractice application bundles.
Supports multiple platforms and packaging methods.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    try:
        import PyInstaller
        print("PyInstaller already installed")
        return True
    except ImportError:
        print("Installing PyInstaller...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install PyInstaller")
            return False

def create_spec_file():
    """Create PyInstaller spec file for MusicPractice."""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icons/*.png', 'icons'),
        ('third-party', 'third-party'),
    ],
    hiddenimports=[
        'librosa',
        'sounddevice',
        'soundfile',
        'numpy',
        'scipy',
        'PySide6',
        'demucs',
        'torch',
        'vamp',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MusicPractice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/icon_256x256.png',
)
'''

    with open('MusicPractice.spec', 'w') as f:
        f.write(spec_content)
    print("Created MusicPractice.spec")

def build_with_pyinstaller():
    """Build the application using PyInstaller."""
    if not install_pyinstaller():
        return False

    create_spec_file()

    print("Building application with PyInstaller...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "MusicPractice.spec"
        ])
        print("Build completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False

def create_macos_app_bundle():
    """Create a proper macOS .app bundle."""
    if sys.platform != 'darwin':
        print("macOS app bundle creation only available on macOS")
        return False

    app_name = "MusicPractice.app"
    app_path = Path("dist") / app_name
    contents_path = app_path / "Contents"
    macos_path = contents_path / "MacOS"
    resources_path = contents_path / "Resources"

    # Create directory structure
    macos_path.mkdir(parents=True, exist_ok=True)
    resources_path.mkdir(parents=True, exist_ok=True)

    # Create Info.plist
    info_plist = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>MusicPractice</string>
    <key>CFBundleIdentifier</key>
    <string>com.musicpractice.app</string>
    <key>CFBundleName</key>
    <string>MusicPractice</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
</dict>
</plist>'''

    with open(contents_path / "Info.plist", 'w') as f:
        f.write(info_plist)

    # Copy executable
    exe_path = Path("dist") / "MusicPractice"
    if exe_path.exists():
        shutil.copy2(exe_path, macos_path / "MusicPractice")
        os.chmod(macos_path / "MusicPractice", 0o755)

    # Copy resources
    for resource in ["icons", "third-party"]:
        item = Path(resource)
        if item.exists():
            shutil.copytree(item, resources_path / item.name, dirs_exist_ok=True)

    # Create icon
    if Path("icons/icon_512x512.png").exists():
        os.system(f"sips -s format icns icons/icon_512x512.png --out {resources_path}/AppIcon.icns")

    print(f"Created macOS app bundle: {app_path}")
    return True

def main():
    """Main build function."""
    print("MusicPractice Application Builder")
    print("=" * 40)

    if len(sys.argv) > 1 and sys.argv[1] == "--macos":
        # macOS-specific build
        if build_with_pyinstaller():
            create_macos_app_bundle()
    else:
        # Standard PyInstaller build
        build_with_pyinstaller()

    print("\nBuild options:")
    print("  python build_app.py          # Standard build")
    print("  python build_app.py --macos  # macOS .app bundle")

    print("\n" + "=" * 60)
    print("IMPORTANT: Vamp Plugins HIGHLY RECOMMENDED")
    print("=" * 60)
    print("✅ The bundled app works fully without Vamp plugins")
    print("🎯 BUT we STRONGLY encourage installing them for much better accuracy!")
    print("")
    print("Core features (always available):")
    print("  - Chord detection, key estimation, beat tracking")
    print("")
    print("Professional enhancements (highly recommended):")
    print("  - Chordino (nnls-chroma) - Much more accurate chords")
    print("  - QM Key Detector (qm-vamp-plugins) - Better key detection")
    print("  - QM BarBeat Tracker (qm-vamp-plugins) - Precise beat tracking")
    print("")
    print("Installation instructions are in the README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()