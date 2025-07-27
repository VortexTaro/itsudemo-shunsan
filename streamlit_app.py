# ... existing code ...
from datetime import datetime
import base64

# --- デザイン設定 ---
# 画像のURL
bg_image_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/59c1c5a9-d232-4861-a477-a8726e632759/d15q6x0-438c8235-513c-43f1-9b88-1516e13ca40f.jpg/v1/fill/w_1024,h_1280,q_75,strp/fantasy_sky_bg_02_by_joannastar_stock_d15q6x0-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJI"

custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,700;1,400&display=swap');

/* Main app background */
.stApp {{
    background-image: url("{bg_image_url}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* App title */
h1 {{
    font-family: 'Lora', serif;
    color: #EAEAEA;
    text-shadow: 2px 2px 8px #000;
}}

/* General text & Chat Bubbles */
body, .st-emotion-cache-10trblm, div[data-testid="stChatMessage"] > div > div > p {{
    color: #EAEAEA;
    font-family: 'Lora', serif;
    text-shadow: 1px 1px 3px #000, -1px -1px 3px #000, 1px -1px 3px #000, -1px 1px 3px #000;
}}

/* Chat Bubbles Container */
div[data-testid="stChatMessage"] {{
    border-radius: 15px;
    padding: 1.2em;
    border: 1px solid rgba(255, 255, 255, 0.15);
    background-color: rgba(10, 20, 30, 0.8); /* Darker, less transparent */
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}}

/* Chat Input Box */
div[data-testid="stChatInput"] {{
    background-color: transparent;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
}}

textarea[data-testid="stChatInputTextArea"] {{
    background-color: rgba(10, 20, 30, 0.8);
    color: #EAEAEA;
    font-family: 'Lora', serif;
    border-radius: 15px;
    border: 1px solid rgba(0, 255, 255, 0.4);
    transition: all 0.3s ease;
}}

textarea[data-testid="stChatInputTextArea"]:focus {{
    border-color: rgba(0, 255, 255, 0.9);
    box-shadow: 0 0 12px rgba(0, 255, 255, 0.6);
}}

/* Expander for source files */
div[data-testid="stExpander"] {{
    border-color: rgba(255, 255, 255, 0.15);
    background-color: rgba(10, 20, 30, 0.75);
    border-radius: 15px;
}}

summary[data-testid="stExpanderHeader"] {{
    color: #EAEAEA;
    font-family: 'Lora', serif;
    text-shadow: 1px 1px 3px #000;
}}

/* Spinner text color */
.stSpinner > div > div {{
    color: #EAEAEA;
    text-shadow: 1px 1px 3px #000;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- 初期設定 ---
st.title("いつでもしゅんさん")
# ... existing code ...
