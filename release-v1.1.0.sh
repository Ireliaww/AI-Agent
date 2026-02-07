#!/bin/bash
# AI-Agent v1.1.0 GitHub Release Script
# Professional version tagging and push workflow

set -e  # Exit on error

echo "🚀 AI-Agent v1.1.0 Release Workflow"
echo "===================================="
echo ""

# Navigate to project root
cd "/Users/ericwang/LLM Agent/AI-Agent"

# Disable git pager for this session
export GIT_PAGER=cat

echo "📁 Current directory: $(pwd)"
echo ""

# 1. Check git status
echo "1️⃣ Checking git status..."
git status
echo ""

# 2. Add all changes
echo "2️⃣ Adding all changes..."
git add .
echo "   ✅ Files staged"
echo ""

# 3. Show what will be committed
echo "3️⃣ Files to be committed:"
git diff --cached --name-status
echo ""

# 4. Commit changes
echo "4️⃣ Creating commit..."
git commit -m "Release v1.1.0: Bug fixes and dual-mode research agent

Major Updates:
- Fixed code extraction regex bug
- Fixed method naming (generate_content_async → generate_content)  
- Added dual-mode capability to EnhancedResearchAgent
- Implemented smart paper title extraction
- Updated architecture diagram to v2
- Created professional CHANGELOG and release notes

Bug Fixes:
- Code extraction now properly removes markdown code blocks
- CodingAgent integration fixed  
- Import paths corrected
- All LLM method calls updated

Documentation:
- Updated README with v1.1.0 features
- Created CHANGELOG.md
- Added detailed release notes in docs/releases/
- Updated task.md with all completions

See CHANGELOG.md and docs/releases/v1.1.0.md for full details.
"
echo "   ✅ Commit created"
echo ""

# 5. Create annotated tag
echo "5️⃣ Creating version tag..."
git tag -a v1.1.0 -m "Version 1.1.0 - Bug Fixes and Enhanced Research Agent

This release fixes critical bugs and adds dual-mode research capabilities.

Key Improvements:
- Dual-mode EnhancedResearchAgent (paper analysis + general research)
- Smart paper title extraction from natural language
- Fixed code generation markdown extraction bug
- Corrected all API method calls
- Professional CHANGELOG and release documentation

Status: Production Ready ✅
All Critical Bugs Fixed
"
echo "   ✅ Tag v1.1.0 created"
echo ""

# 6. Show tags
echo "6️⃣ Current tags:"
git tag -l
echo ""

# 7. Push to GitHub
echo "7️⃣ Pushing to GitHub..."
echo "   Branch: $(git branch --show-current)"
read -p "   Press ENTER to push, or Ctrl+C to cancel..."
git push origin $(git branch --show-current)
echo "   ✅ Code pushed"
echo ""

# 8. Push tags
echo "8️⃣ Pushing tags..."
git push origin v1.1.0
# Or push all tags: git push origin --tags
echo "   ✅ Tags pushed"
echo ""

# 9. Summary
echo "✨ Release Complete!"
echo "==================="
echo ""
echo "📦 Version: v1.1.0"
echo "🌿 Branch: $(git branch --show-current)"
echo "🔖 Tag: v1.1.0"
echo ""
echo "Next Steps:"
echo "1. Visit GitHub and create a release from tag v1.1.0"
echo "2. Copy content from docs/releases/v1.1.0.md to release notes"
echo "3. Mark as 'Latest Release'"
echo ""
echo "GitHub Release URL (after creating):"
echo "https://github.com/Ireliaww/AI-Agent/releases/tag/v1.1.0"
echo ""
echo "🎉 Done!"
