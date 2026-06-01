#!/bin/bash
set -e

# Configuration
PRIVATE_REPO_DIR=$(pwd)
PUBLIC_REPO_DIR="../fulloch"  # Path to your local public repo clone

# Version — pass as argument, enter at prompt, or skip prompt for date default
VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    read -rp "Version [e.g. 2.1.0, Enter to use 2.0.$(date +'%Y%m%d')]: " VERSION
    VERSION="${VERSION:-"2.0.$(date +'%Y%m%d')"}"
fi
TAG="v${VERSION}"
COMMIT_MSG="Release ${TAG}: $(date +'%Y-%m-%d')"

echo "Publishing ${TAG}..."

mkdir -p "$PUBLIC_REPO_DIR"

# 1. Check public repo is clean
cd "$PUBLIC_REPO_DIR" || exit 1
if [[ -n $(git status -s) ]]; then
    echo "Error: Public repo has uncommitted changes. Please clean it first."
    exit 1
fi

# 2. Check tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: Tag ${TAG} already exists in public repo."
    exit 1
fi

# 3. Clean out old public files (except .git)
find . -maxdepth 1 -not -name '.git' -not -name '.' -exec rm -rf {} +

# 4. Export clean snapshot from private repo
cd "$PRIVATE_REPO_DIR" || exit 1
# --worktree-attributes honours .gitattributes export-ignore rules (e.g. CLAUDE.md)
git archive --worktree-attributes HEAD | tar -x -C "$PUBLIC_REPO_DIR"

echo "✅ Repo files transferred"

# 5. Stamp the version into the HACS integration manifest so HACS shows the
#    correct version and can notify users of updates.
MANIFEST="$PUBLIC_REPO_DIR/custom_components/fulloch/manifest.json"
if [[ -f "$MANIFEST" ]]; then
    # Replace "version": "<anything>" with the new version (portable sed)
    sed -i.bak "s/\"version\": \"[^\"]*\"/\"version\": \"${VERSION}\"/" "$MANIFEST"
    rm -f "${MANIFEST}.bak"
    echo "✅ manifest.json version → ${VERSION}"
fi

# 6. Commit and push public repo
cd "$PUBLIC_REPO_DIR" || exit 1
git add .
git commit -m "$COMMIT_MSG"
git push origin main

echo "✅ Public repo updated"

# 7. Create and push the release tag — HACS uses these to detect new versions
#    and notify users. The tag must exist on the remote before HACS picks it up.
git tag -a "$TAG" -m "Fulloch ${TAG}"
git push origin "$TAG"

echo "✅ Tagged ${TAG} and pushed"

# 8. Create a GitHub release (requires gh CLI)
if command -v gh &>/dev/null; then
    gh release create "$TAG" \
        --title "Fulloch ${TAG}" \
        --notes "Release ${TAG} — see README for changes." \
        --latest
    echo "✅ GitHub release created: ${TAG}"
else
    echo "⚠️  gh CLI not found — create the GitHub release manually at:"
    echo "   https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/releases/new?tag=${TAG}"
fi

echo ""
echo "Done. HACS users will see the update once the release is live."
