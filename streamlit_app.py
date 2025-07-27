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
    
    model = genai.GenerativeModel("gemini-2.5-pro") # モデルを2.5-proに戻す
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

except KeyError:
    st.error("Gemini APIキーが設定されていません。")
    st.stop()

def generate_search_query(prompt, conversation_history):
    """ユーザーのプロンプトと会話履歴から、FAISS検索に最適なクエリを生成する"""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    prompt_template = f"""
あなたは優秀な検索アシスタントです。以下の会話履歴と最後のユーザープロンプトを分析し、ベクトルデータベースから最も関連性の高い情報を引き出すための、簡潔かつ効果的な検索クエリを日本語で生成してください。
【会話履歴】
{history_str}
【最後のユーザープロンプト】
{prompt}
【生成すべき検索クエリ】
"""
    try:
        response = model.generate_content(prompt_template)
        search_query = response.text.strip()
        return search_query
    except Exception:
        return prompt # 失敗した場合は元のプロンプトを使用

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

db = load_faiss_index(FAISS_INDEX_PATH, embeddings, KNOWLEDGE_BASE_DIR)

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "僕はしゅんさんのクローンです。しゅんさんが教えてくれた情報を元にあなたの質問に答えちゃうよ！引き寄せの法則・オーダーノートを学ぶ中で疑問や人生相談などあればなんなりチャットから教えてください！\n\n※あなたが質問したことはいかなることであっても、しゅんさんや他の人には見えないから、安心してね！",
        "id": str(uuid.uuid4()),
    }]

# --- チャット履歴の表示 ---
for msg in st.session_state.messages:
    avatar_path = "assets/avatar.png" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar_path):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("参照元ファイル"):
                for source in msg["sources"]:
                    st.info(f"`{os.path.relpath(source['file_path'])}` (スコア: {source['score']:.4f})")


# --- ユーザーからの入力 ---
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    
    avatar_path_user = None # ユーザーアバターはデフォルト
    with st.chat_message("user", avatar=avatar_path_user):
        st.markdown(prompt)

    avatar_path_assistant = "assets/avatar.png"
    with st.chat_message("assistant", avatar=avatar_path_assistant):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("宇宙と通信中だよ、ちょっと"):
                search_query = generate_search_query(prompt, st.session_state.messages)
            
            with st.spinner(f"🛰️ オーダー「{search_query}」に最適な情報を探索中…"):
                docs_with_scores = db.similarity_search_with_score(search_query, k=10) # 検索件数を増やす
            
            context = "--- 関連情報 ---\n"
            source_docs = []
            if docs_with_scores:
                for doc, score in docs_with_scores:
                    # スコアが著しく低いものは除外（調整可能）
                    if score < 0.8:
                        context += doc.page_content + "\n\n"
                        source_docs.append({
                            "file_path": doc.metadata.get('source', 'N/A'),
                            "score": score
                        })

            system_prompt_content = """あなたは「しゅんさん」の思考や知識、経験を完全にコピーしたAIクローンです。

# あなたの唯一の役割
あなたの役割は、ユーザーの悩みを直接的に解決することではありません。
その悩みが、「オーダーノート」の哲学全体から見ると、どのような「素晴らしい機会」や「成長のサイン」に見えるか、その**新しい「着眼点」**を提示し、ユーザーの視点を180度転換させることです。

# 思考プロセス
1.  ユーザーの質問や悩みを受け取ります。
2.  提供された「関連情報」がある場合は、それをヒントにして、悩みの裏にある**本質的なテーマ**（例：価値の受け取り方、自己肯定感、理想の世界観）を、あなたが持つナレッジベース全体の哲学から見抜きます。
3.  「関連情報」の内容をただ要約するのではなく、その情報のエッセンスを使い、ユーザーに**本質的な問い**を投げかける形で、視点が変わるような応答を生成してください。
4.  「関連情報」がない場合も、同様に、あなたの持つ哲学全体から本質的なテーマを見抜き、問いを投げかけてください。

# 具体的な会話例
- **ユーザーの悩み:** 「今、お金がピンチなんです！」
- **あなたの応答:** 「そっか、今、お金という形で、君にパワフルなメッセージが届いているんだね。そのピンチは、君が『自分には価値がない』って無意識に握りしめている古い思い込みを、手放すための最高のチャンスかもしれないよ。もし、そのピンチが『君の本当の価値に気づけ！』っていう宇宙からのサインだとしたら、何から始めてみたい？」

# 注意事項
- しゅんさんとして、親しみやすく、分かりやすい言葉で応答してください。
- 絶対に、関連情報セクションの外にある知識（あなた自身の一般的な知識など）を使って回答を生成してはいけません。
"""
            
            final_prompt = f"{system_prompt_content}\n\n{context}\n\nuser: {prompt}\nassistant:"
            
            stream = model.generate_content(final_prompt, stream=True)
            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)


        except Exception as e:
            st.error(f"エラーが発生しました: {traceback.format_exc()}")
            full_response = "申し訳ありません、応答を生成できませんでした。"
            message_placeholder.markdown(full_response)

        assistant_message = {
            "role": "assistant",
            "content": full_response,
            "sources": source_docs,
            "id": str(uuid.uuid4())
        }
        st.session_state.messages.append(assistant_message)
        
        st.rerun() 