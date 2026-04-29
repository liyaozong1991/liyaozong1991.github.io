#!/bin/bash

PROJECT_DIR="/home/li.yaozong/work_space/github/liyaozong1991.github.io"
PAPER_DIR="${PROJECT_DIR}/_posts/论文阅读"
LOG_DIR="${PROJECT_DIR}/tools"
LOG_FILE="${LOG_DIR}/daily_paper.log"
TODAY=$(date +%Y-%m-%d)

export CURSOR_API_KEY="crsr_21df98ea579372e510ca0428cfa41ba8759dc8352a6f0bdfa04edc382941e5c3"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log "===== 开始执行每日论文阅读任务 ====="

cd "$PROJECT_DIR" || { log "ERROR: 无法进入项目目录"; exit 1; }

git pull origin master >> "$LOG_FILE" 2>&1

EXISTING_TITLES=$(ls "$PAPER_DIR"/*.md 2>/dev/null | xargs -I{} basename {} .md | sed 's/^[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-//')
log "已有文章标题: ${EXISTING_TITLES}"

PROMPT=$(cat <<'PROMPT_EOF'
你是一个推荐系统领域的论文阅读助手。请完成以下任务：

## 任务
在当前项目的 `_posts/论文阅读/` 目录下，撰写一篇国内互联网大厂（字节跳动、阿里巴巴、美团、腾讯、百度、快手、京东等）在推荐系统方向上的论文阅读笔记。

## 要求

### 选题
1. 论文必须是国内互联网大厂在推荐系统方向上的工作，发表或预印本时间为2025年或2026年
2. 优先选择 arXiv 上有预印本的论文，可以在 arXiv 上搜索
3. 方向可以涵盖：CTR预估、排序模型、召回模型、序列推荐、生成式推荐、Semantic ID、多任务学习、多场景建模、特征交互、用户行为建模、重排序、预排序、冷启动、多模态推荐、LLM+推荐、Agent推荐等
4. **绝对不能与已有文章重复**

### 已有文章（不要写这些主题）
PROMPT_EOF
)

PROMPT="${PROMPT}
${EXISTING_TITLES}

### 文件命名格式
\`${TODAY}-论文简短中文标题.md\`

### 文档格式
严格遵循以下 front matter 和正文格式：

\`\`\`markdown
---
categories: [机器学习]
tags: [推荐系统, 其他相关标签]
math: true
title: 论文中文标题
---

**论文**: 论文英文标题
**链接**: [arXiv链接](arXiv链接)
**机构**: 发表机构
**时间**: 发表时间

## 1. 问题背景

（描述论文要解决的核心问题，2-3段）

## 2. 方法概述

（概述论文提出的方法框架）

### 2.1 子方法1

（详细技术描述，包含关键公式，用 LaTeX）

### 2.2 子方法2

（详细技术描述）

## 3. 实验

（关键实验结果，可用表格展示）

## 4. 总结与思考

（总结论文贡献，加入个人思考）
\`\`\`

### 内容质量要求
1. 深入理解论文核心思想，不要只是翻译摘要
2. 关键公式用 LaTeX 书写（用 \$\$ 包裹行间公式，\$ 包裹行内公式）
3. 总结部分要有自己的思考和评价
4. 文章总长度在 2000-5000 字之间
5. tags 标签根据论文实际内容设置，都用小写

请搜索并选择一篇合适的论文，然后在 _posts/论文阅读/ 目录下创建对应的 markdown 文件。"

cd "$PROJECT_DIR"

log "开始调用 cursor-agent 生成论文笔记..."

cursor-agent \
    -p "$PROMPT" \
    --model "claude-4.6-opus-high-thinking" \
    --output-format text \
    --trust \
    >> "$LOG_FILE" 2>&1

AGENT_EXIT_CODE=$?
log "cursor-agent 退出码: ${AGENT_EXIT_CODE}"

if [ $AGENT_EXIT_CODE -ne 0 ]; then
    log "ERROR: cursor-agent 执行失败"
    exit 1
fi

NEW_FILES=$(find "$PAPER_DIR" -name "${TODAY}-*.md" -newer "$LOG_FILE" 2>/dev/null || find "$PAPER_DIR" -name "${TODAY}-*.md" 2>/dev/null)
if [ -z "$NEW_FILES" ]; then
    log "WARNING: 未检测到新生成的文件"
else
    log "新生成的文件: ${NEW_FILES}"
fi

cd "$PROJECT_DIR"
git add -A >> "$LOG_FILE" 2>&1
git commit -m "feat: 每日论文阅读 ${TODAY} - 自动生成" >> "$LOG_FILE" 2>&1
git push origin master >> "$LOG_FILE" 2>&1

PUSH_EXIT_CODE=$?
if [ $PUSH_EXIT_CODE -eq 0 ]; then
    log "Git push 成功"
else
    log "WARNING: Git push 失败，退出码: ${PUSH_EXIT_CODE}"
fi

log "===== 每日论文阅读任务完成 ====="
