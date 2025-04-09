#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display usage
usage() {
    echo "Usage: $0 [major|minor|patch] [--release] [--message <commit-message>]"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    usage
fi

# Extract version segment, optional release flag, and optional commit message
VERSION_SEGMENT=$1
RELEASE_FLAG=$2
COMMIT_MESSAGE=$3

# Default commit message if none is provided
if [ -z "$COMMIT_MESSAGE" ]; then
    COMMIT_MESSAGE="Bump version ($VERSION_SEGMENT)"
fi

# Update version using hatch
echo "Updating version ($VERSION_SEGMENT)..."
hatch version $VERSION_SEGMENT

# Commit the version bump
echo "Committing version bump..."
git add .
git commit -m "$COMMIT_MESSAGE"

# Push changes to GitHub
echo "Pushing changes to GitHub..."
git push

# If the --release flag is provided, create a release tag
if [ "$RELEASE_FLAG" == "--release" ]; then
    NEW_VERSION=$(hatch version) # Get the new version
    echo "Creating release tag v$NEW_VERSION..."
    git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"
    git push origin "v$NEW_VERSION"
fi

echo "Done."
