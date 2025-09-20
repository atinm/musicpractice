#!/usr/bin/env python3
"""
Convert existing PNG icons to platform-specific formats (ICO, ICNS).
Uses the icons in the icons/ directory.
"""

import os
import sys
from pathlib import Path

def convert_icons():
    """Convert PNG icons to ICO and ICNS formats."""
    icons_dir = Path("icons")

    if not icons_dir.exists():
        print("Icons directory not found!")
        return False

    # Find the largest PNG icon
    png_icons = list(icons_dir.glob("icon_*.png"))
    if not png_icons:
        print("No icon PNG files found in icons/ directory!")
        return False

    # Use the largest icon (512x512 or 256x256)
    largest_icon = None
    for size in [512, 256, 128, 64, 32, 16]:
        icon_file = icons_dir / f"icon_{size}x{size}.png"
        if icon_file.exists():
            largest_icon = icon_file
            break

    if not largest_icon:
        print("No suitable icon found!")
        return False

    print(f"Using icon: {largest_icon}")

    try:
        from PIL import Image

        # Create ICO for Windows
        try:
            img = Image.open(largest_icon)
            # Create ICO with multiple sizes
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save('icon.ico', sizes=sizes)
            print("✓ Created icon.ico for Windows")
        except Exception as e:
            print(f"✗ Failed to create ICO: {e}")

        # Create ICNS for macOS
        try:
            img = Image.open(largest_icon)
            # Create ICNS with multiple sizes
            sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
            img.save('icon.icns', sizes=sizes)
            print("✓ Created icon.icns for macOS")
        except Exception as e:
            print(f"✗ Failed to create ICNS: {e}")

        # Copy the main PNG for Linux
        try:
            import shutil
            shutil.copy2(largest_icon, 'icon.png')
            print("✓ Created icon.png for Linux")
        except Exception as e:
            print(f"✗ Failed to copy PNG: {e}")

        return True

    except ImportError:
        print("PIL/Pillow not available. Install with: pip install Pillow")
        print("For now, you can manually use the icons from the icons/ directory:")
        print(f"  - Use {largest_icon} as your application icon")
        return False

def show_icon_info():
    """Show information about available icons."""
    icons_dir = Path("icons")

    if not icons_dir.exists():
        print("No icons directory found!")
        return

    print("Available icons:")
    png_icons = sorted(icons_dir.glob("icon_*.png"))
    for icon in png_icons:
        size = icon.stem.split('_')[1]  # Extract size from filename
        print(f"  - {icon.name} ({size})")

    print(f"\nTotal: {len(png_icons)} icon files")

def main():
    """Main function."""
    print("MusicPractice Icon Converter")
    print("=" * 30)

    show_icon_info()
    print()

    if convert_icons():
        print("\n✓ Icon conversion completed!")
        print("You can now build the application with:")
        print("  python3 build_app.py")
    else:
        print("\n✗ Icon conversion failed!")
        print("Make sure you have icons in the icons/ directory")

if __name__ == "__main__":
    main()
