# ... existing code ...
import json
from datetime import datetime
import base64

# デザイン定義（この部分が欠落していた）
bg_image_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/4d99d253-1524-492c-a86a-72a397a6a759/d8vohc4-c22f183a-23a9-43c2-9092-23b6b66e33a4.png/v1/fill/w_1280,h_4496,q_80,strp/haunted_library_ref_by_gooseworx_d8vohc4-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIZCJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NDQ5NiIsInBhdGgiOiJcL2ZcLzRkOTlkMjUzLTE1MjQtNDkyYy1hODZhLTcyYTM5N2E2YTc1OVwvZDh2b2hjNC1jMjJmMTgzYS0yM2E5LTQzYzItOTA5Mi0yM2I2YjY2ZTMzYTQucG5nIiwid2lkdGgiOiI8PTEyODAifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.YtB2U_z2g5K_Dbk2JpB4c62Q2z2iK285d3I01e_4a_E"
custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
.stApp {{
    background-image: url("{bg_image_url}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
h1 {{
    font-family: 'Noto Sans JP', sans-serif;
    color: #FFFFFF;
    text-shadow: 2px 2px 8px #000000;
}}
div[data-testid="stChatMessage"] {{
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background-color: rgba(240, 240, 245, 0.9);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}}
div[data-testid="stChatMessage"] > div > div > p {{
    color: #1E1E1E;
    font-family: 'Noto Sans JP', sans-serif;
    text-shadow: none;
}}
div[data-testid="stChatInput"] {{
    background-color: rgba(15, 15, 15, 0.8);
    border-top: 1px solid rgba(255, 255, 255, 0.2);
}}
textarea[data-testid="stChatInputTextArea"] {{
    background-color: rgba(230, 230, 235, 1);
    color: #1E1E1E;
    font-family: 'Noto Sans JP', sans-serif;
    border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.2);
}}
textarea[data-testid="stChatInputTextArea"]:focus {{
    border-color: #0078FF;
    box-shadow: 0 0 8px #0078FF;
}}
div[data-testid="stExpander"] {{
    border-color: rgba(0, 0, 0, 0.2);
    background-color: rgba(220, 220, 225, 0.95);
    border-radius: 12px;
}}
summary[data-testid="stExpanderHeader"] > div p {{
    color: #1E1E1E;
    text-shadow: none;
}}
div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {{
    color: #1E1E1E;
}}
div[data-testid="stExpander"] textarea {{
    color: #1E1E1E;
    background-color: rgba(255, 255, 255, 0.7);
}}
.stSpinner > div > div {{
    color: #FFFFFF;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# --- 初期設定 ---
st.title("いつでもしゅんさん")
# ... existing code ...
