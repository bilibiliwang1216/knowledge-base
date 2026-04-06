import os
import json
from notion_client import Client

# 配置
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "33682958b1f4803e9dc1cb0066f28195")
OUTPUT_FILE = "knowledge_data.json"

notion = Client(auth=NOTION_TOKEN)

def get_page_content(page_id):
    """获取页面正文内容"""
    blocks = notion.blocks.children.list(block_id=page_id)
    content = []
    for block in blocks["results"]:
        block_type = block["type"]
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
            texts = block[block_type].get("rich_text", [])
            text = "".join([t["plain_text"] for t in texts])
            if text:
                content.append(text)
        elif block_type == "code":
            texts = block["code"].get("rich_text", [])
            text = "".join([t["plain_text"] for t in texts])
            if text:
                content.append(f"[代码]\n{text}")
        elif block_type == "bulleted_list_item":
            texts = block["bulleted_list_item"].get("rich_text", [])
            text = "".join([t["plain_text"] for t in texts])
            if text:
                content.append(f"• {text}")
        elif block_type == "numbered_list_item":
            texts = block["numbered_list_item"].get("rich_text", [])
            text = "".join([t["plain_text"] for t in texts])
            if text:
                content.append(f"- {text}")
        elif block_type == "image":
            content.append("[图片]")
    return "\n".join(content)

def sync():
    print("开始同步 Notion 知识库...")
    results = []

    # 查询数据库所有条目
    response = notion.databases.query(database_id=DATABASE_ID)
    pages = response["results"]

    # 处理分页
    while response.get("has_more"):
        response = notion.databases.query(
            database_id=DATABASE_ID,
            start_cursor=response["next_cursor"]
        )
        pages.extend(response["results"])

    print(f"共找到 {len(pages)} 条记录")

    for page in pages:
        props = page["properties"]

        # 获取标题
        title = ""
        if props.get("标题") and props["标题"].get("title"):
            title = "".join([t["plain_text"] for t in props["标题"]["title"]])

        # 获取分类
        category = ""
        if props.get("分类") and props["分类"].get("select"):
            category = props["分类"]["select"]["name"]

        # 获取标签
        tags = []
        if props.get("标签") and props["标签"].get("multi_select"):
            tags = [t["name"] for t in props["标签"]["multi_select"]]

        # 获取页面正文
        content = get_page_content(page["id"])

        if title:  # 只保存有标题的条目
            results.append({
                "id": page["id"],
                "title": title,
                "category": category,
                "tags": tags,
                "content": content,
                "url": page["url"]
            })
            print(f"  已同步: {title}")

    # 保存到本地文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n同步完成！共 {len(results)} 条，已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    sync()
