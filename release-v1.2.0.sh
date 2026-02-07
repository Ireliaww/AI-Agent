#!/bin/bash

# Release script for v1.2.0 - AI Thinking Process Visualization
# Major feature release

set -e

VERSION="v1.2.0"
RELEASE_TYPE="Major Feature Release"

echo "🚀 Starting release process for ${VERSION}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check directory
if [ ! -d "deep-research-agent" ]; then
    echo "❌ Error: Must run from AI-Agent root directory"
    exit 1
fi

echo "✅ Directory check passed"

# Check for uncommitted changes
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

# Verify new files
echo ""
echo "📝 Verifying new files..."

NEW_FILES=(
    "deep-research-agent/utils/artifact_manager.py"
    "deep-research-agent/utils/progress_display.py"
    "deep-research-agent/tests/test_artifacts.py"
    "deep-research-agent/tests/demo_progress_display.py"
    "docs/releases/v1.2.0.md"
)

for file in "${NEW_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found"
        exit 1
    fi
    echo "✅ $file exists"
done

# Run tests
echo ""
echo "🧪 Running tests..."

cd deep-research-agent/tests

# Test artifact generation
echo "Testing artifact generation..."
python test_artifacts.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Artifact generation tests passed"
else
    echo "❌ Artifact generation tests failed"
    exit 1
fi

# Test progress display
echo "Testing progress display..."
python demo_progress_display.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Progress display demo passed"
else
    echo "❌ Progress display demo failed"
    exit 1
fi

cd ../..

# Git operations
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
git commit -m "Release ${VERSION} - AI Thinking Process Visualization

Major Features:
- ✨ Complete AI thinking process artifact generation (5 markdown files)
- ✨ Real-time progress display with Rich library
- ✨ Step-by-step visual feedback with AI thinking panels
- ✨ Project structure tree visualization
- ✨ Syntax-highlighted code previews
- ✨ Reproduction report generation

New Files:
- utils/artifact_manager.py - Artifact generation system
- utils/progress_display.py - Progress visualization
- tests/test_artifacts.py - Artifact tests
- tests/demo_progress_display.py - Visual demo

Modified:
- agents/enhanced_research_agent.py - Artifact saving
- agents/enhanced_coding_agent.py - Progress display
- agents/coordinator.py - Orchestration

Impact: MAJOR - Transforms paper reproduction into transparent, 
educational experience with complete AI reasoning visualization."

echo "✅ Changes committed"

# Create tag
echo ""
echo "🏷️  Creating git tag ${VERSION}..."

git tag -a "${VERSION}" -m "Release ${VERSION} - ${RELEASE_TYPE}

AI Thinking Process Visualization

This major release introduces complete transparency into the AI's 
reasoning process during paper reproduction:

## Artifacts Generated (5 files per reproduction):
- 01_PAPER_ANALYSIS.md - RAG-enhanced understanding
- 02_UNDERSTANDING.md - Key insights & decisions  
- 03_ARCHITECTURE_DESIGN.md - Component design
- 04_IMPLEMENTATION_LOG.md - Generation timeline
- REPRODUCTION_REPORT.md - Complete summary

## Real-Time Progress Display:
- Step headers with progress bars
- AI thinking panels  
- Project structure trees
- Syntax-highlighted code previews
- Statistics tables
- Completion summaries

Tested and verified. No breaking changes."

echo "✅ Tag created"

# Push
echo ""
echo "📤 Ready to push to GitHub"
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
    echo "⚠️  Skipped push. You can push manually:"
    echo "   git push origin main"
    echo "   git push origin ${VERSION}"
fi

# Next steps
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Release ${VERSION} prepared successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://github.com/Ireliaww/AI-Agent/releases/new"
echo "2. Select tag: ${VERSION}"
echo "3. Title: ${VERSION} - AI Thinking Process Visualization"
echo "4. Copy content from: docs/releases/v1.2.0.md"
echo "5. Mark as 'Latest release'"
echo "6. Publish release"
echo ""
echo "📝 Release notes at: docs/releases/v1.2.0.md"
echo ""
echo "🎉 Great work on this major feature release!"
