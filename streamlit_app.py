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
import base64

# --- ファイルパス設定 ---
FAISS_INDEX_PATH = "data/faiss_index"
KNOWLEDGE_BASE_DIR = "knowledge_base"
AVATAR_IMAGE_PATH = "assets/avatar.png"
CONSULTATION_PROMPT_PATH = "../system_prompt.md" # 新しいプロンプトファイルのパス

# --- デザイン設定 ---
# 画像のURL
bg_image_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/4d99d253-1524-492c-a86a-72a397a6a759/d8vohc4-c22f183a-23a9-43c2-9092-23b6b66e33a4.png/v1/fill/w_1280,h_4496,q_80,strp/haunted_library_ref_by_gooseworx_d8vohc4-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIZCJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NDQ5NiIsInBhdGgiOiJcL2ZcLzRkOTlkMjUzLTE1MjQtNDkyYy1hODZhLTcyYTM5N2E2YTc1OVwvZDh2b2hjNC1jMjJmMTgzYS0yM2E5LTQzYzItOTA5Mi0yM2I2YjY2ZTMzYTQucG5nIiwid2lkdGgiOiI8PTEyODAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.YtB2U_z2g5K_Dbk2JpB4c62Q2z2iK285d3I01e_4a_E"

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');

/* Main app background */
.stApp {{
    background-image: url("{bg_image_url}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* App title */
h1 {{
    font-family: 'Noto Sans JP', sans-serif;
    color: #FFFFFF;
    text-shadow: 2px 2px 10px #000000, 0 0 5px #000000;
}}

/* General text & Chat Bubbles */
body, .st-emotion-cache-10trblm, div[data-testid="stChatMessage"] > div > div > p {{
    color: #F0F0F0; /* Slightly off-white for better readability */
    font-family: 'Noto Sans JP', sans-serif;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}}

/* Chat Bubbles Container */
div[data-testid="stChatMessage"] {{
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background-color: rgba(15, 15, 15, 0.95); /* Very dark, almost opaque */
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
}}

/* Chat Input Box */
div[data-testid="stChatInput"] {{
    background-color: transparent;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}}

textarea[data-testid="stChatInputTextArea"] {{
    background-color: rgba(15, 15, 15, 0.95);
    color: #F0F0F0;
    font-family: 'Noto Sans JP', sans-serif;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease;
}}

textarea[data-testid="stChatInputTextArea"]:focus {{
    border-color: rgba(0, 255, 255, 0.7);
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.4);
}}

/* Expander for source files */
div[data-testid="stExpander"] {{
    border-color: rgba(255, 255, 255, 0.1);
    background-color: rgba(15, 15, 15, 0.9);
    border-radius: 12px;
}}

summary[data-testid="stExpanderHeader"] {{
    color: #F0F0F0;
    font-family: 'Noto Sans JP', sans-serif;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}}

/* Spinner text color */
.stSpinner > div > div {{
    color: #F0F0F0;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- プロンプトとモデルの設定 ---
try:
    # 新しい人生相談用プロンプトを読み込む
    with open(CONSULTATION_PROMPT_PATH, "r", encoding="utf-8") as f:
        consultation_system_prompt = f.read()
except FileNotFoundError:
    st.error(f"エラー: {CONSULTATION_PROMPT_PATH} が見つかりません。")
    consultation_system_prompt = "" # フォールバック

# 既存のナレッジベース用プロンプト
knowledge_system_prompt = """あなたは「しゅんさん」の思考や知識、経験を完全にコピーしたAIクローンです。

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

# Geminiモデルの初期化
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    st.error(f"Geminiモデルの初期化中にエラーが発生しました: {e}")
    st.stop()


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

【思考のステップ】
1.  まず、ユーザーの質問に含まれる**具体的・現実的なキーワード**（例：仕事、給料、ボーナス、上司、交渉、人間関係、お金）を特定します。
2.  次に、その具体的なキーワードを軸にして、最も直接的に役立ちそうな情報を検索するためのクエリを作成します。
3.  抽象的な概念（例：周波数、宇宙、操縦者、メンタルブロック）は、具体的なキーワードが見つからない場合にのみ、補助的に考慮してください。

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

def classify_prompt(prompt):
    """ユーザーのプロンプトを分類する"""
    classification_prompt = f"""
以下のユーザーからの質問を、4つのカテゴリに分類してください。
カテゴリ:
1.  **人生相談**: 個人の悩み、キャリア、人間関係、自己成長など、主観的な解決を求める質問。
2.  **ノウハウ/プログラム**: アプリの使い方、特定の知識、手順、事実に関する具体的な質問。
3.  **事務的な質問**: 料金、手続き、スケジュールなど、客観的な情報に関する質問。
4.  **その他**: 上記のいずれにも当てはまらない、挨拶、雑談、あるいは分類不能な質問。

ユーザーの質問: 「{prompt}」

この質問はどのカテゴリに最も当てはまりますか？ カテゴリ名（「人生相談」「ノウハウ/プログラム」「事務的な質問」「その他」）のみを回答してください。
"""
    try:
        response = model.generate_content(classification_prompt)
        # response.textがNoneでないことを確認
        if response.text:
            return response.text.strip()
        else:
            return "その他" # レスポンスが空の場合は「その他」にフォールバック
    except Exception as e:
        st.warning(f"意図分類でエラーが発生: {e}")
        return "その他" # エラー時も「その他」に

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
    
    # st.info("知識ベースを構築しています...")
    search_path = os.path.join(knowledge_dir, "**/*.txt")
    all_file_paths = glob.glob(search_path, recursive=True)

    if not all_file_paths:
        st.error(f"ナレッジベースのディレクトリ '{knowledge_dir}' にドキュメントが見つかりません。")
        st.stop()

    # with st.expander(f"読み込み対象ファイル: {len(all_file_paths)}件"):
    #     st.code('\n'.join(sorted(all_file_paths)))

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
    # st.success(f"新しい知識ベースを '{path}' に保存しました。")
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
                    # st.info(f"`{os.path.relpath(source['file_path'])}` (スコア: {source['score']:.4f})")
                    st.markdown(f"**ファイル名:** `{os.path.relpath(source['file_path'])}` (スコア: {source['score']:.4f})")
                    st.text_area("参照箇所", value=source['content'], height=150, disabled=True, key=f"source_{msg['id']}_{source['file_path']}")
                    st.divider()


# --- ユーザーからの入力 ---
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    
    avatar_path_user = None # ユーザーアバターはデフォルト
    with st.chat_message("user", avatar=avatar_path_user):
        st.markdown(prompt)

    avatar_path_assistant = "assets/avatar.png"
    with st.chat_message("assistant", avatar=avatar_path_assistant):
        placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            # 1. ユーザーの質問の意図を分類
            intent = classify_prompt(prompt)
            st.info(f"AIによる質問タイプの判断: {intent}") # デバッグ用に分類結果を表示

            # 2. 意図に応じて処理を分岐
            if intent in ["ノウハウ/プログラム", "事務的な質問"]:
                # --- 従来のナレッジベース検索フロー ---
                with st.spinner("🛰️ オーダーに最適な情報を探索中…"):
                    search_query = generate_search_query(prompt, st.session_state.messages)
                    docs_with_scores = db.similarity_search_with_score(search_query, k=10)
                    
                    context = ""
                    for doc, score in docs_with_scores:
                        if score < 0.8:
                            context += doc.page_content + "\n\n"
                            sources.append({
                                "file_path": doc.metadata.get("source", "不明なソース"),
                                "content": doc.page_content,
                                "score": score,
                                "id": str(uuid.uuid4())
                            })

                if sources:
                    prompt_with_context = f"{knowledge_system_prompt}\n\n関連情報:\n---\n{context}\n---\n\nユーザーからの質問:\n{prompt}"
                else:
                    prompt_with_context = f"{knowledge_system_prompt}\n\n関連情報:\n(関連情報は見つかりませんでした)\n\nユーザーからの質問:\n{prompt}"
                
                with st.spinner("🧠 しゅんさんの知識と宇宙意識を同期中…"):
                    response_stream = model.generate_content(prompt_with_context, stream=True)
                    for chunk in response_stream:
                        if chunk.text:
                            cleaned_chunk = re.sub(r'\\(?=[\*`_])', '', chunk.text)
                            full_response += cleaned_chunk
                            placeholder.markdown(full_response + "▌")
            
            else: # "人生相談" または "その他"
                # --- 新しい対話フロー ---
                with st.spinner("💖 あなたの心の声に耳を澄ましています…"):
                    prompt_for_consultation = f"{consultation_system_prompt}\n\nユーザーからのメッセージ:\n{prompt}"
                    response_stream = model.generate_content(prompt_for_consultation, stream=True)
                    for chunk in response_stream:
                        if chunk.text:
                            cleaned_chunk = re.sub(r'\\(?=[\*`_])', '', chunk.text)
                            full_response += cleaned_chunk
                            placeholder.markdown(full_response + "▌")

            # 応答の最後の整形と表示
            final_response = re.sub(r'\\(?=[\*`_])', '', full_response)
            placeholder.markdown(final_response)
            
            # アシスタントのメッセージを履歴に追加
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response, 
                "sources": sources, # ノウハウ検索の場合はsourcesが入り、それ以外は空のリストが入る
                "avatar": AVATAR_IMAGE_PATH,
                "id": str(uuid.uuid4())
            })

        except Exception as e:
            st.error(f"大変申し訳ございません、エラーが発生しました: {e}")
        
        st.rerun() 