import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import asyncio
import uuid
import re
import traceback
import json
from datetime import datetime

# --- 初期設定 ---
st.title("いつでもしゅんさん")

# StreamlitのsecretsからAPIキーを設定
try:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        raise KeyError("API key not found")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-1.5-pro-latest") # 安定性を優先してモデルを一旦変更
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

except KeyError:
    st.error("Gemini APIキーが設定されていません。")
    st.stop()

# --- 知識ベース構築 ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_INDEX_PATH = "data/faiss_index"

@st.cache_resource
def load_faiss_index(path, _embeddings, knowledge_dir):
    if os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss")):
        try:
            return FAISS.load_local(path, _embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"既存DBの読み込みに失敗: {e}。再構築します。")
    
    st.info("知識ベースを構築しています...")
    search_path = os.path.join(knowledge_dir, "**/*.txt")
    all_file_paths = glob.glob(search_path, recursive=True)

    if not all_file_paths:
        st.error(f"ナレッジベースのディレクトリ '{knowledge_dir}' にドキュメントが見つかりません。")
        st.stop()

    with st.expander(f"読み込み対象ファイル: {len(all_file_paths)}件"):
        st.code('\n'.join(sorted(all_file_paths)))

    documents = []
    for file_path in all_file_paths:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"ファイル読み込みエラー: {file_path}, {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    db = FAISS.from_documents(chunks, _embeddings)
    os.makedirs(path, exist_ok=True)
    db.save_local(path)
    st.success(f"新しい知識ベースを '{path}' に保存しました。")
    return db

# --- ナレッジベースの配置 ---
# st.info("ナレッジベースの準備をしています...")
# 古いプロジェクトから新しいプロジェクトへナレッジベースをコピー
# os.system(f"cp -r オーダーノート現実創造プログラム/* hikiyose_app_fresh/knowledge_base/")

db = load_faiss_index(FAISS_INDEX_PATH, embeddings, KNOWLEDGE_BASE_DIR)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "僕はしゅんさんのクローンです。オーダーノートや引き寄せについて、何でも聞いてね！",
        "id": str(uuid.uuid4()),
    }]

# --- チャット履歴の表示 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- ユーザーからの入力 ---
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            docs_with_scores = db.similarity_search_with_score(prompt, k=5)
            
            context = "--- 関連情報 ---\n"
            source_docs = []
            if docs_with_scores:
                for doc, score in docs_with_scores:
                    context += doc.page_content + "\n\n"
                    source_docs.append({
                        "file_path": doc.metadata.get('source', 'N/A'),
                        "score": score
                    })

            # プロンプトファイルの作成と読み込み
            prompt_file_path = "docs/system_prompt.md"
            system_prompt_content = """あなたは「しゅんさん」の思考や知識、経験を完全にコピーしたAIクローンです。
しゅんさんとして、親しみやすく、分かりやすい言葉で、ユーザーの悩みや質問に答えてください。
提供された「関連情報」を最優先の知識ベースとし、その情報のみに基づいて回答を生成してください。
情報が不足している場合は、正直に「その質問については、僕の知識の中には情報がないみたい。ごめんね！」と答えてください。
絶対に、関連情報以外の知識（あなた自身の一般的な知識など）を使って回答を生成してはいけません。
"""
            with open(prompt_file_path, "w", encoding="utf-8") as f:
                f.write(system_prompt_content)
            
            final_prompt = f"{system_prompt_content}\n\n{context}\n\nuser: {prompt}\nassistant:"
            
            response = model.generate_content(final_prompt)
            full_response = response.text
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"エラーが発生しました: {traceback.format_exc()}")
            full_response = "申し訳ありません、応答を生成できませんでした。"
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
        if source_docs:
            with st.expander("参照元ファイル"):
                for source in source_docs:
                    st.info(f"`{os.path.relpath(source['file_path'])}` (スコア: {source['score']:.4f})")
        
        st.rerun() 