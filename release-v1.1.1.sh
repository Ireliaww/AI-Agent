#!/bin/bash

# Release script for v1.1.1 - Bug Fix Release
# This script automates the release process

set -e  # Exit on error

VERSION="v1.1.1"
RELEASE_TYPE="Bug Fix"

echo "🚀 Starting release process for ${VERSION}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1: Check we're in the right directory
if [ ! -d "deep-research-agent" ]; then
    echo "❌ Error: Must run from AI-Agent root directory"
    exit 1
fi

echo "✅ Directory check passed"

# Step 2: Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Release cancelled"
        exit 1
    fi
fi

# Step 3: Verify key files were modified
echo ""
echo "📝 Verifying critical fixes..."

FILES_TO_CHECK=(
    "deep-research-agent/rag/vector_store/gemini_embedding.py"
    "deep-research-agent/rag/vector_store/chroma_store.py"
    "deep-research-agent/agents/coordinator.py"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found"
        exit 1
    fi
    
    # Check for gemini-embedding-001
    if [[ "$file" == *"gemini_embedding.py" ]]; then
        if grep -q "gemini-embedding-001" "$file"; then
            echo "✅ $file - embedding model updated"
        else
            echo "❌ Error: $file missing gemini-embedding-001"
            exit 1
        fi
    fi
    
    # Check for os import
    if [[ "$file" == *"coordinator.py" ]]; then
        if grep -q "^import os" "$file"; then
            echo "✅ $file - os import added"
        else
            echo "❌ Error: $file missing os import"
            exit 1
        fi
    fi
done

# Step 4: Update version in main files
echo ""
echo "📝 Updating version numbers..."

# Update main.py if it has version
if grep -q "VERSION" deep-research-agent/main.py; then
    sed -i '' "s/VERSION = .*/VERSION = \"${VERSION}\"/" deep-research-agent/main.py
    echo "✅ Updated main.py version"
fi

# Step 5: Update CHANGELOG
echo ""
echo "📝 Updating CHANGELOG.md..."

CHANGELOG_ENTRY="
## [${VERSION}] - $(date +%Y-%m-%d)

### Fixed
- 🐛 **Critical**: Fixed RAG system embedding API (changed to gemini-embedding-001)
- 🐛 Added missing \`import os\` in coordinator agent
- ✅ Project files now save correctly to disk
- ✅ Embeddings generate successfully without 404 errors
- ✅ Paper understanding now uses RAG-retrieved content

### Changed
- Updated embedding model from text-embedding-004 to gemini-embedding-001

### Technical
- Modified: \`rag/vector_store/gemini_embedding.py\`
- Modified: \`rag/vector_store/chroma_store.py\`
- Modified: \`agents/coordinator.py\`
- Modified: \`agents/enhanced_coding_agent.py\`
"

if [ -f "CHANGELOG.md" ]; then
    # Insert after ## [Unreleased]
    sed -i '' "/## \[Unreleased\]/a\\
${CHANGELOG_ENTRY}
" CHANGELOG.md
    echo "✅ CHANGELOG.md updated"
else
    echo "⚠️  CHANGELOG.md not found, skipping"
fi

# Step 6: Git operations
echo ""
echo "📦 Preparing git commit..."

git add -A

echo ""
echo "Files to be committed:"
git status --short

echo ""
read -p "Proceed with commit? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Release cancelled"
    exit 1
fi

# Commit
git commit -m "Release ${VERSION} - Critical bug fixes

- Fixed RAG embedding API (gemini-embedding-001)
- Fixed missing os import in coordinator
- Fixed project file saving
- Verified end-to-end paper reproduction

Closes #XX (if applicable)"

echo "✅ Changes committed"

# Step 7: Create git tag
echo ""
echo "🏷️  Creating git tag ${VERSION}..."

git tag -a "${VERSION}" -m "Release ${VERSION} - ${RELEASE_TYPE}

Critical Fixes:
- RAG System: Fixed embedding model (404 error resolved)
- File Saving: Added missing os import
- Project Generation: Now saves to disk correctly

Tested with 'Attention Is All You Need' paper reproduction."

echo "✅ Tag created"

# Step 8: Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
echo "This will push:"
echo "  - Commits to main branch"
echo "  - Tag ${VERSION}"
echo ""

read -p "Push to GitHub now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin main
    git push origin "${VERSION}"
    echo "✅ Pushed to GitHub"
else
    echo "⚠️  Skipped push. You can push manually with:"
    echo "   git push origin main"
    echo "   git push origin ${VERSION}"
fi

# Step 9: Display next steps
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Release ${VERSION} prepared successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://github.com/Ireliaww/AI-Agent/releases/new"
echo "2. Select tag: ${VERSION}"
echo "3. Title: ${VERSION} - Critical Bug Fixes"
echo "4. Copy content from: docs/releases/v1.1.1.md"
echo "5. Publish release"
echo ""
echo "📝 Release notes at: docs/releases/v1.1.1.md"
echo ""
echo "🎉 Great work!"
