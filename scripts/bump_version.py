#!/usr/bin/env python3
"""
Simple version bumper - only edits __init__.py
"""

import re
import subprocess
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


def create_git_tag(version):
    """Create and push git tag"""
    tag = f"v{version}"

    try:
        # Check if tag already exists
        result = subprocess.run(
            f"git tag -l {tag}", shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            print(f"⚠️  Tag {tag} already exists locally. Deleting and recreating...")
            subprocess.run(f"git tag -d {tag}", shell=True, check=True)

        # Create tag
        subprocess.run(f"git tag {tag}", shell=True, check=True)
        print(f"✅ Created git tag {tag}")

        # Push tag
        subprocess.run(f"git push origin {tag}", shell=True, check=True)
        print(f"✅ Pushed tag {tag} to remote")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating/pushing tag: {e}")
        print("You may need to create the tag manually:")
        print(f"  git tag {tag}")
        print(f"  git push origin {tag}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py <major|minor|patch>")
        print("\nThis updates __init__.py, commits, and creates a git tag.")
        print("Then create a GitHub release using the created tag.")
        sys.exit(1)

    bump_type = sys.argv[1]
    if bump_type not in ["major", "minor", "patch"]:
        print("Error: must be major|minor|patch")
        sys.exit(1)

    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)

    print(f"Current: {current_version}")
    print(f"New: {new_version}")

    # Update version in __init__.py
    update_version(new_version)

    # Commit the change
    try:
        subprocess.run("git add src/bayescurvefit/__init__.py", shell=True, check=True)
        subprocess.run(
            f"git commit -m 'Bump version to {new_version}'", shell=True, check=True
        )
        print(f"✅ Committed version bump to {new_version}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error committing: {e}")
        print("Please commit manually:")
        print("  git add src/bayescurvefit/__init__.py")
        print(f"  git commit -m 'Bump version to {new_version}'")
        return

    # Create and push git tag
    create_git_tag(new_version)

    print(f"\n🎉 Version {new_version} is ready!")
    print(f"Next step: Create GitHub release with tag v{new_version}")
    print("Go to: https://github.com/ndu-bioinfo/BayesCurveFit/releases")


if __name__ == "__main__":
    main()
