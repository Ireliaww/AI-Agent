# 🚀 发布指南

## 文件结构

```
AI-Agent/
├── CHANGELOG.md                    # 版本更新日志（简洁版）
├── README.md                       # 主文档（已更新v1.1.0）
├── release-v1.1.0.sh              # 自动发布脚本
├── docs/
│   └── releases/
│       ├── v1.0.0.md              # v1.0.0 详细发布说明
│       └── v1.1.0.md              # v1.1.0 详细发布说明
└── images/
    ├── paper_reproduction_architecture.png        # v1.0架构图
    └── paper_reproduction_architecture_v2.png     # v1.1架构图
```

---

## 📋 已准备的文件

### 1. CHANGELOG.md
- 位置：项目根目录
- 格式：遵循 [Keep a Changelog](https://keepachangelog.com/)
- 内容：v1.1.0 和 v1.0.0 的简洁更新列表

### 2. Release Notes
- `docs/releases/v1.1.0.md` - 详细的v1.1.0更新说明
- `docs/releases/v1.0.0.md` - 详细的v1.0.0发布说明

### 3. 更新的文档
- `README.md` - 已更新v1.1.0信息和新架构图
- `images/paper_reproduction_architecture_v2.png` - 新架构图

---

## 🎯 发布步骤

### 方法1: 使用自动脚本（推荐）

```bash
cd "/Users/ericwang/LLM Agent/AI-Agent"
./release-v1.1.0.sh
```

脚本会自动：
1. ✅ 检查git状态
2. ✅ Add所有更改
3. ✅ 显示将要提交的文件
4. ✅ 创建commit（包含详细说明）
5. ✅ 创建annotated tag `v1.1.0`
6. ✅ Push到GitHub
7. ✅ Push tags

---

### 方法2: 手动执行

```bash
cd "/Users/ericwang/LLM Agent/AI-Agent"

# 1. 查看状态
git status

# 2. 添加所有更改
git add .

# 3. 查看将要提交的文件
git diff --cached --name-status

# 4. 提交
git commit -m "Release v1.1.0: Bug fixes and dual-mode research agent

Major Updates:
- Fixed code extraction regex bug
- Fixed method naming (generate_content_async → generate_content)  
- Added dual-mode capability to EnhancedResearchAgent
- Implemented smart paper title extraction
- Updated architecture diagram to v2
- Created professional CHANGELOG and release notes

See CHANGELOG.md for full details.
"

# 5. 创建tag
git tag -a v1.1.0 -m "Version 1.1.0 - Bug Fixes and Enhanced Research Agent

Key Improvements:
- Dual-mode EnhancedResearchAgent
- Smart paper title extraction
- Fixed code generation bugs
- Professional CHANGELOG

Status: Production Ready ✅
"

# 6. Push
git push origin main  # 或你的分支名
git push origin v1.1.0
# 或推送所有tags: git push origin --tags
```

---

## 📝 在GitHub上创建Release

Push完成后：

1. **访问**: https://github.com/Ireliaww/AI-Agent/releases

2. **点击**: "Draft a new release"

3. **选择tag**: v1.1.0

4. **Release标题**: `v1.1.0 - Bug Fixes & Dual-Mode Research Agent`

5. **描述**: 复制 `docs/releases/v1.1.0.md` 的内容

6. **勾选**: "Set as the latest release"

7. **发布**: Click "Publish release"

---

## ✅ 检查清单

发布前确认：

- [x] CHANGELOG.md 已创建并包含v1.0.0和v1.1.0
- [x] docs/releases/v1.0.0.md 已创建
- [x] docs/releases/v1.1.0.md 已创建
- [x] README.md 已更新到v1.1.0
- [x] 新架构图已添加到images/
- [x] release-v1.1.0.sh 脚本已创建并可执行
- [ ] Git commit已创建
- [ ] Git tag v1.1.0 已创建
- [ ] 代码已push到GitHub
- [ ] Tag已push到GitHub
- [ ] GitHub Release已创建

---

## 🎨 专业标准

本项目遵循：

- **[Keep a Changelog](https://keepachangelog.com/)** - CHANGELOG格式
- **[Semantic Versioning](https://semver.org/)** - 版本号规则
  - `1.1.0` = MAJOR.MINOR.PATCH
  - Bug fixes → PATCH
  - New features (backwards compatible) → MINOR  
  - Breaking changes → MAJOR

---

## 📞 问题排查

### 如果push失败

```bash
# 查看remote
git remote -v

# 如果没有remote，添加
git remote add origin https://github.com/Ireliaww/AI-Agent.git

# 重新push
git push -u origin main
```

### 如果tag已存在

```bash
# 删除本地tag
git tag -d v1.1.0

# 删除远程tag
git push origin :refs/tags/v1.1.0

# 重新创建
git tag -a v1.1.0 -m "..."
git push origin v1.1.0
```

---

## 🎉 完成！

发布完成后，你的GitHub仓库将有：

- ✅ 清晰的版本历史（CHANGELOG.md）
- ✅ 详细的发布说明（docs/releases/）
- ✅ Git tags标记每个版本
- ✅ GitHub Releases页面展示所有版本
- ✅ 专业的项目文档结构

访问你的项目：https://github.com/Ireliaww/AI-Agent
