import json
import os
import subprocess
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_FILE = "knowledge_data.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

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
            result = subprocess.run(
                ["python", "sync.py"],
                capture_output=True, text=True, encoding="utf-8"
            )
            st.text(result.stdout)
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
