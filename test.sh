#!/bin/bash

# 脚本说明：批量注释指定目录下所有文件中包含 "http" 的行（添加 # 注释）
# 使用方法：./comment_http_lines.sh <目标目录> [文件扩展名过滤（可选）]

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [file_extension_filter]"
    exit 1
fi

TARGET_DIR="$1"
EXT_FILTER="bak"  # 默认匹配所有文件

# 遍历目录下的所有文件（支持递归）
find "$TARGET_DIR" -type f -name "*.$EXT_FILTER" | while read -r FILE; do
    echo "Processing: $FILE"
    rm $FILE
    
    # 备份原文件（添加 .bak 后缀）
    ## cp -n "$FILE" "$FILE.bak"
    #
    #sed -i '' '/http/ s/^\(.*\)$/<!-- \1 -->/' "$FILE"
    #
    ## 检查 sed 操作是否成功
    #if [ $? -eq 0 ]; then
    #    echo "注释完成，备份文件：$FILE.bak"
    #else
    #    echo "处理失败：$FILE" >&2
    #fi
done

echo "所有文件处理完成。"
