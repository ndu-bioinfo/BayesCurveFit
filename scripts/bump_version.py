#!/usr/bin/env python3
"""
Simple version bumper - only edits __init__.py
"""

import re
import sys
from pathlib import Path


def get_current_version():
    """Get current version from __init__.py"""
    init_path = Path("src/bayescurvefit/__init__.py")
    content = init_path.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    if match:
        return match.group(1)
    return "0.0.0"


def bump_version(version, bump_type):
    """Bump version based on type"""
    parts = version.split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")

    return f"{major}.{minor}.{patch}"


def update_version(new_version):
    """Update version in __init__.py"""
    init_path = Path("src/bayescurvefit/__init__.py")
    content = init_path.read_text()

    new_content = re.sub(
        r'__version__ = "[^"]*"', f'__version__ = "{new_version}"', content
    )

    init_path.write_text(new_content)
    print(f"✅ Updated __init__.py to version {new_version}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <major|minor|patch>")
        print(
            "\nThis updates only __init__.py. Then create a GitHub release with tag v<version>"
        )
        sys.exit(1)

    bump_type = sys.argv[1]
    if bump_type not in ["major", "minor", "patch"]:
        print("Error: must be major|minor|patch")
        sys.exit(1)

    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)

    print(f"Current: {current_version}")
    print(f"New: {new_version}")

    update_version(new_version)
    print("\nNext steps:")
    print("1. git add src/bayescurvefit/__init__.py")
    print(f"2. git commit -m 'Bump version to {new_version}'")
    print("3. git push")
    print(f"4. Create GitHub release with tag v{new_version}")


if __name__ == "__main__":
    main()
