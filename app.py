import json
import os
import streamlit as st
from notion_client import Client
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_FILE = "knowledge_data.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def get_notion_config():
    try:
        token = st.secrets["NOTION_TOKEN"]
        db_id = st.secrets["NOTION_DATABASE_ID"]
    except Exception:
        token = os.environ.get("NOTION_TOKEN", "")
        db_id = os.environ.get("NOTION_DATABASE_ID", "")
    return token, db_id

def get_page_content(notion, page_id):
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
        elif block_type == "image":
            content.append("[图片]")
    return "\n".join(content)

def sync_from_notion():
    token, db_id = get_notion_config()
    if not token or not db_id:
        st.error("未找到 NOTION_TOKEN 或 NOTION_DATABASE_ID，请检查 Secrets 配置。")
        return False
    notion = Client(auth=token)
    response = notion.databases.query(database_id=db_id)
    pages = response["results"]
    while response.get("has_more"):
        response = notion.databases.query(database_id=db_id, start_cursor=response["next_cursor"])
        pages.extend(response["results"])
    results = []
    for page in pages:
        props = page["properties"]
        title = ""
        if props.get("标题") and props["标题"].get("title"):
            title = "".join([t["plain_text"] for t in props["标题"]["title"]])
        category = ""
        if props.get("分类") and props["分类"].get("select"):
            category = props["分类"]["select"]["name"]
        tags = []
        if props.get("标签") and props["标签"].get("multi_select"):
            tags = [t["name"] for t in props["标签"]["multi_select"]]
        content = get_page_content(notion, page["id"])
        if title:
            results.append({
                "id": page["id"],
                "title": title,
                "category": category,
                "tags": tags,
                "content": content,
                "url": page["url"]
            })
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return len(results)

st.set_page_config(page_title="知识库", page_icon="📚", layout="wide")
st.title("📚 团队知识库")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def build_index(data):
    model = load_model()
    texts = []
    for item in data:
        text = f"{item['title']} {item.get('category', '')} {item.get('content', '')}"
        texts.append(text)
    return model.encode(texts)

def search(query, data, embeddings, top_k=5):
    model = load_model()
    query_embedding = model.encode([query])
    scores = np.dot(embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "title": data[idx]["title"],
            "category": data[idx].get("category", ""),
            "content": data[idx].get("content", ""),
            "url": data[idx].get("url", "")
        })
    return results

# 加载数据
data = load_data()

# 侧边栏
with st.sidebar:
    st.header("操作")
    if st.button("🔄 同步 Notion 数据"):
        with st.spinner("正在同步..."):
            count = sync_from_notion()
            if count is not False:
                st.success(f"同步完成，共 {count} 条")
                st.cache_data.clear()
                st.rerun()

    st.divider()
    st.metric("知识条目总数", len(data))

    # 分类统计
    if data:
        categories = {}
        for item in data:
            cat = item.get("category") or "未分类"
            categories[cat] = categories.get(cat, 0) + 1
        st.subheader("分类统计")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            st.write(f"- {cat}：{count} 条")

# 主区域
if not data:
    st.warning("知识库暂无数据，请点击左侧「同步 Notion 数据」按钮。")
else:
    embeddings = build_index(data)

    query = st.text_input("🔍 输入问题或关键词搜索", placeholder="例如：登录失败怎么办？")

    if query:
        results = search(query, data, embeddings)
        st.subheader(f"搜索结果（共 {len(results)} 条）")

        for r in results:
            with st.expander(f"📄 {r['title']}  |  分类：{r['category'] or '未分类'}  |  相关度：{r['score']:.2f}"):
                if r['content']:
                    st.markdown(r['content'])
                else:
                    st.info("该条目暂无正文内容，点击链接在 Notion 中查看。")
                st.markdown(f"[在 Notion 中打开]({r['url']})")
    else:
        # 未搜索时展示所有条目
        st.subheader("全部知识条目")
        for item in data:
            with st.expander(f"📄 {item['title']}  |  {item.get('category') or '未分类'}"):
                if item.get('content'):
                    st.markdown(item['content'])
                else:
                    st.info("暂无正文内容。")
                st.markdown(f"[在 Notion 中打开]({item['url']})")
