import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_FILE = "knowledge_data.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # 支持中文的多语言模型

# 加载模型（第一次运行会自动下载，约400MB）
print("正在加载模型（首次运行需要下载，请耐心等待）...")
model = SentenceTransformer(MODEL_NAME)
print("模型加载完成！")

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"找不到数据文件 {DATA_FILE}，请先运行 sync.py")
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(data):
    """为所有知识条目生成向量"""
    texts = []
    for item in data:
        # 把标题+分类+内容拼在一起做索引
        text = f"{item['title']} {item.get('category', '')} {item.get('content', '')}"
        texts.append(text)
    print("正在建立搜索索引...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def search(query, data, embeddings, top_k=3):
    """语义搜索，返回最相关的 top_k 条结果"""
    query_embedding = model.encode([query])
    # 计算余弦相似度
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

def main():
    data = load_data()
    if not data:
        return

    print(f"已加载 {len(data)} 条知识条目")
    embeddings = build_index(data)

    print("\n知识库搜索已就绪！输入问题开始搜索，输入 q 退出\n")
    while True:
        query = input("请输入问题：").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        results = search(query, data, embeddings)
        print(f"\n找到 {len(results)} 条相关内容：\n")
        for i, r in enumerate(results, 1):
            print(f"--- 结果 {i} ---")
            print(f"标题：{r['title']}")
            print(f"分类：{r['category']}")
            print(f"相关度：{r['score']:.2f}")
            if r['content']:
                # 只显示前200字
                content_preview = r['content'][:200]
                if len(r['content']) > 200:
                    content_preview += "..."
                print(f"内容：{content_preview}")
            print(f"链接：{r['url']}")
            print()

if __name__ == "__main__":
    main()
