import os
import time
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import requests
import json
from pathlib import Path

# Load .env for local development
load_dotenv()

# ==========================================
# 1. CONFIGURATION & API SETUP
# ==========================================
# When deploying to Streamlit Cloud, add your key to "Secrets".
# For local testing, add GEMINI_API_KEY to your environment or fill .streamlit/secrets.toml.
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or "YOUR_ACTUAL_API_KEY_HERE"
genai.configure(api_key=API_KEY)
DEBUG_MODE = st.secrets.get("DEBUG", False) or os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
model = genai.GenerativeModel('models/gemini-3-flash-preview')

# Page Config
st.set_page_config(
    page_title="The Fund Audit | Mutual Fund Overlap Analyzer",
    page_icon="images/FullLogo_NoBuffer_Large.png",
    layout="wide"
)

# ==========================================
# AMFI NAV DATA MANAGEMENT
AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
AMFI_LOCAL_FILE = Path(__file__).parent / "amfi_nav_data.txt"


def parse_amfi_nav_text(text):
    nav_data = {}
    scheme_names = []

    for line in text.splitlines():
        if not line or 'Scheme Name' in line:
            continue
        parts = [p.strip() for p in line.split(';')]
        if len(parts) < 6:
            continue

        scheme_name = parts[3]
        nav_value = parts[4]
        nav_date = parts[5]

        if not scheme_name:
            continue

        try:
            nav = float(nav_value.replace(',', '').strip())
        except ValueError:
            continue

        key = scheme_name.lower()
        if key not in nav_data:
            nav_data[key] = {
                'nav': nav,
                'date': nav_date,
                'full_name': scheme_name,
                'scheme_code': parts[0]
            }
            scheme_names.append(scheme_name)

    return nav_data, sorted(set(scheme_names))


@st.cache_data(ttl=86400)
def load_amfi_nav_data():
    nav_data = {}
    scheme_names = []
    text = None

    if AMFI_LOCAL_FILE.exists():
        try:
            age = time.time() - AMFI_LOCAL_FILE.stat().st_mtime
            if age < 86400:
                text = AMFI_LOCAL_FILE.read_text("utf-8")
        except Exception:
            text = None

    if text is None:
        try:
            response = requests.get(AMFI_NAV_URL, timeout=20)
            response.raise_for_status()
            text = response.text
            try:
                AMFI_LOCAL_FILE.write_text(text, encoding="utf-8")
                st.success("✓ Downloaded fresh AMFI NAV data")
            except Exception:
                st.warning("Downloaded AMFI NAV data but could not save locally.")
        except Exception as e:
            if AMFI_LOCAL_FILE.exists():
                st.warning(f"Could not download latest AMFI NAV data: {e}. Using cached local copy.")
                text = AMFI_LOCAL_FILE.read_text("utf-8")
            else:
                st.error(f"Could not load AMFI NAV data: {e}")
                return nav_data, scheme_names

    try:
        nav_data, scheme_names = parse_amfi_nav_text(text)
    except Exception as e:
        st.error(f"Error parsing AMFI data: {e}")

    return nav_data, scheme_names


@st.cache_data(ttl=86400)
def get_nav_from_local_data(fund_name, nav_data):
    if not fund_name or not nav_data:
        return None

    search_key = fund_name.lower().strip()
    if search_key in nav_data:
        return nav_data[search_key]

    for key, value in nav_data.items():
        if search_key in key or key in search_key:
            return value
    return None

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

# Load and encode logo image
import base64

logo_path = Path(__file__).parent / "images/FullLogo_Transparent_NoBuffer.png"
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_data}" style="width: 80px; height: 80px; margin-right: 15px; vertical-align: middle;">'
else:
    logo_html = '📊'

# Custom CSS for a cleaner "FinTech" look
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    body, .stApp, .main, .stText, .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, .stSelectbox, .stTextInput {
        color: #000000 !important;
    }
    .metric-card {
        border-radius: 24px;
        padding: 22px 24px;
        color: #ffffff;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08);
        height: 170px;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        overflow: hidden;
    }
    .metric-card h3 {
        margin: 0 0 12px;
        font-size: 18px;
        letter-spacing: 0.02em;
        opacity: 0.92;
    }
    .metric-card .value {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
        line-height: 1.05;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-card .subtitle {
        margin: 10px 0 0;
        font-size: 14px;
        opacity: 0.78;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-card.card-a { background: linear-gradient(135deg, #7da6ff 0%, #6a84ff 100%); }
    .metric-card.card-b { background: linear-gradient(135deg, #43d19f 0%, #1a8f6a 100%); }
    .metric-card.card-c { background: linear-gradient(135deg, #f06682 0%, #b3344f 100%); }
    .metric-card.card-d { background: linear-gradient(135deg, #9f7df3 0%, #5e4ce0 100%); }
    .stMetric, .stMetric * {
        color: #111 !important;
    }
    .modal-backdrop {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .modal-content {
        background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 100%);
        border-radius: 22px;
        padding: 32px 32px 24px 32px;
        max-width: 560px;
        width: min(92vw, 560px);
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.14);
        text-align: center;
        animation: slideIn 0.32s ease-out;
        border: 1px solid rgba(59, 130, 246, 0.14);
        position: relative;
        overflow: hidden;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-25px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .modal-content h2 {
        color: #1d4ed8;
        margin-bottom: 16px;
        font-size: 30px;
    }
    .modal-content p {
        color: #334155;
        line-height: 1.7;
        margin-bottom: 18px;
        font-size: 15px;
    }
    .modal-content ul {
        text-align: left;
        color: #475569;
        margin: 20px auto 0 auto;
        display: inline-block;
        padding-left: 0;
        list-style: none;
        max-width: 520px;
    }
    .modal-content ul li {
        margin-bottom: 12px;
        line-height: 1.6;
        padding-left: 24px;
        position: relative;
    }
    .modal-content ul li:before {
        content: "✔";
        position: absolute;
        left: 0;
        top: 0;
        color: #2563eb;
        font-weight: 700;
    }
    .modal-button-shell {
        width: 100%;
        max-width: 520px;
        margin: 24px auto 0 auto;
    }
    .modal-content form {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-top: 16px;
    }
    .modal-content .stButton > button {
        width: 100% !important;
        max-width: 520px !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 0 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.2) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        display: block !important;
    }
    .modal-content .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 18px 36px rgba(37, 99, 235, 0.28) !important;
    }
    .modal-accent {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 8px;
        background: linear-gradient(90deg, #2563eb 0%, #60a5fa 50%, #93c5fd 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #60A5FA 0%, #93C5FD 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 6px 20px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        min-height: 38px !important;
        max-height: 38px !important;
        box-shadow: 0 8px 20px rgba(96, 165, 250, 0.2) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 12px 28px rgba(96, 165, 250, 0.35) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HEADER SECTION
# ==========================================
st.markdown(f"""
<div style="background: linear-gradient(135deg, #FFEB99 0%, #FFB84D 100%); padding: 30px 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 10px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
    <h1 style="font-size: 2.8em; margin-bottom: 10px; font-weight: 700;">{logo_html}The Fund Audit</h1>
    <p style="font-size: 1.2em; margin-bottom: 15px; opacity: 0.9;">Identify hidden stock overlap between your mutual funds</p>
    <p style="font-size: 1em; line-height: 1.5; max-width: 700px; margin: 0 auto;">Over-diversification often leads to 'closet indexing'—where you pay active fees for passive returns. Enter two funds to see how much they actually differ.</p>
    <div style="margin-top: 20px;">
        <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px; font-size: 0.85em;">🔍 Smart Analysis</span>
        <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px; font-size: 0.85em;">📈 AI-Powered</span>
        <span style="background: rgba(255,255,255,0.2); padding: 6px 12px; border-radius: 15px; margin: 0 8px; font-size: 0.85em;">⚡ Instant Results</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. INPUT SECTION
# ==========================================
nav_data, amfi_funds = load_amfi_nav_data()

if amfi_funds:
    fund_options = ["Select a fund..."] + amfi_funds
else:
    fund_options = ["Select a fund..."]

col1, col2, col3 = st.columns([1, 1, 0.5])

with col1:
    fund_a_choice = st.selectbox(
        "Enter Fund A",
        options=fund_options,
        help="Start typing to search popular fund names."
    )
    fund_a = "" if fund_a_choice == "Select a fund..." else fund_a_choice

with col2:
    fund_b_choice = st.selectbox(
        "Enter Fund B",
        options=fund_options,
        help="Start typing to search popular fund names."
    )
    fund_b = "" if fund_b_choice == "Select a fund..." else fund_b_choice

with col3:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    button_clicked = st.button("Compare", type="primary", use_container_width=True)

# ==========================================
# 4. ANALYSIS LOGIC
# ==========================================
if button_clicked:
    if fund_a and fund_b:
        with st.spinner(f"Analyzing {fund_a} vs {fund_b}..."):
            
            # The System Prompt ensures the AI returns structured JSON
            prompt = f"""
            Act as a professional financial data analyst. 
            For these two Indian mutual funds: '{fund_a}' and '{fund_b}', provide the current top 10 stock holdings and their approximate percentage weights.
            
            Calculate the Portfolio Overlap Percentage (the sum of the lower weight of any common stock).
            
            Also provide current NAV (Net Asset Value) for both funds.
            
            IMPORTANT: For the "insight" field, provide a brief 2-sentence educational analysis that:
            - Does NOT recommend buying, selling, or investing in any fund
            - Does NOT give financial advice or investment recommendations
            - Focuses purely on analytical observations about the overlap and diversification
            - Clearly states this is for educational purposes only
            
            CRITICAL: Your response must be ONLY a valid JSON object. Do not include any text before or after the JSON. Do not use markdown formatting. Do not add explanations or comments.
            
            Format the response strictly as a JSON object with this structure:
            {{
                "fund_a_name": "actual name",
                "fund_b_name": "actual name",
                "fund_a_holdings": {{"Stock Name": weight, ...}},
                "fund_b_holdings": {{"Stock Name": weight, ...}},
                "overlap_percentage": value,
                "common_stocks": ["Stock A", "Stock B", ...],
                "fund_a_nav": current_nav_price,
                "fund_b_nav": current_nav_price,
                "insight": "A brief 2-sentence educational analysis focusing on overlap and diversification, with no investment recommendations"
            }}
            Return ONLY the JSON object, without any surrounding markdown, code fences, explanation, or extra text.
            """
            
            raw_response = ""
            clean_response = ""
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        top_p=0.8,
                        max_output_tokens=2048,
                    )
                )
                raw_response = getattr(response, "text", str(response))
                
                # More robust JSON extraction
                clean_response = raw_response.strip()
                
                # Remove markdown code blocks if present
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                
                clean_response = clean_response.strip()
                
                # Try to find JSON object if there's extra text
                import re
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    clean_response = json_match.group(0)
                
                data = json.loads(clean_response)

                # --- Display Results ---
                overlap = data.get('overlap_percentage', 0)
                fund_a_holdings = data.get('fund_a_holdings', {}) or {}
                fund_b_holdings = data.get('fund_b_holdings', {}) or {}
                fund_a_name = data.get('fund_a_name', fund_a)
                fund_b_name = data.get('fund_b_name', fund_b)
                nav_a = get_nav_from_local_data(fund_a_name, nav_data) or get_nav_from_local_data(fund_a, nav_data)
                nav_b = get_nav_from_local_data(fund_b_name, nav_data) or get_nav_from_local_data(fund_b, nav_data)
                fund_a_nav = nav_a['nav'] if nav_a else data.get('fund_a_nav', 'N/A')
                fund_b_nav = nav_b['nav'] if nav_b else data.get('fund_b_nav', 'N/A')

                df_a = pd.DataFrame(list(fund_a_holdings.items()), columns=['Stock', 'Weight'])
                df_b = pd.DataFrame(list(fund_b_holdings.items()), columns=['Stock', 'Weight'])
                df_a = df_a.sort_values('Weight', ascending=False).reset_index(drop=True)
                df_b = df_b.sort_values('Weight', ascending=False).reset_index(drop=True)

                common_df = pd.merge(df_a, df_b, on='Stock', how='inner', suffixes=('_A', '_B'))
                if not common_df.empty:
                    common_df['OverlapWeight'] = common_df[['Weight_A', 'Weight_B']].min(axis=1)
                    common_df = common_df.sort_values('OverlapWeight', ascending=False).reset_index(drop=True)

                common_count = len(common_df)
                top_common = ", ".join(common_df['Stock'].head(5).tolist()) if common_count else "None"

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                metric_col1.markdown(f"""
                    <div class='metric-card card-a'>
                        <div>
                            <h3>Total Portfolio Overlap</h3>
                            <p class='value'>{overlap}%</p>
                        </div>
                        <p class='subtitle'>{'High redundancy' if overlap > 30 else 'Moderate overlap' if overlap > 15 else 'Clean diversification'}</p>
                    </div>
                """, unsafe_allow_html=True)

                metric_col2.markdown(f"""
                    <div class='metric-card card-b'>
                        <div>
                            <h3>Common Holdings</h3>
                            <p class='value'>{common_count}</p>
                        </div>
                        <p class='subtitle'>Shared stocks between the two portfolios</p>
                    </div>
                """, unsafe_allow_html=True)

                metric_col3.markdown(f"""
                    <div class='metric-card card-c'>
                        <div>
                            <h3>Top Shared Stocks</h3>
                            <p class='value'>{top_common}</p>
                        </div>
                        <p class='subtitle'>Most important common holdings</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

                if overlap > 30:
                    st.error("⚠️ High Redundancy Detected")
                elif overlap > 15:
                    st.warning("⚠️ Moderate Overlap")
                else:
                    st.success("✅ Clean Diversification")

                st.info(f"**Expert Insight:** {data.get('insight', 'No insight available.')}" )

                holdings_tab, overlap_tab, nav_tab = st.tabs(["Fund Holdings", "Overlap Summary", "NAV"])

                with holdings_tab:
                    def get_font_size(name: str) -> str:
                        length = len(name)
                        if length > 120:
                            return "14px"
                        if length > 90:
                            return "16px"
                        if length > 70:
                            return "18px"
                        if length > 50:
                            return "20px"
                        return "24px"

                    header_container_style = "min-height: 60px; display: flex; align-items: flex-start; margin-bottom: 16px;"
                    header_style = "display:block; width:100%; white-space: normal; overflow-wrap: anywhere; word-break: break-word; line-height: 1.2;"
                    font_a = get_font_size(fund_a_name)
                    font_b = get_font_size(fund_b_name)

                    holdings_col1, holdings_col2 = st.columns(2)
                    with holdings_col1:
                        st.markdown(
                            f"<div style='{header_container_style}'><h3 style='font-size: {font_a}; {header_style}'>{fund_a_name}</h3></div>",
                            unsafe_allow_html=True
                        )
                        if not df_a.empty:
                            st.dataframe(df_a, use_container_width=True)
                            fig_a = px.pie(df_a.head(10), values='Weight', names='Stock', title=f"Top 10 Holdings: {fund_a_name}", hole=0.4)
                            st.plotly_chart(fig_a, use_container_width=True)
                        else:
                            st.info("No holdings data available for Fund A.")

                    with holdings_col2:
                        st.markdown(
                            f"<div style='{header_container_style}'><h3 style='font-size: {font_b}; {header_style}'>{fund_b_name}</h3></div>",
                            unsafe_allow_html=True
                        )
                        if not df_b.empty:
                            st.dataframe(df_b, use_container_width=True)
                            fig_b = px.pie(df_b.head(10), values='Weight', names='Stock', title=f"Top 10 Holdings: {fund_b_name}", hole=0.4)
                            st.plotly_chart(fig_b, use_container_width=True)
                        else:
                            st.info("No holdings data available for Fund B.")

                with overlap_tab:
                    st.write("### Overlap Calculation")
                    st.write("Overlap is calculated as the sum of the lower weight for every stock held by both funds.")
                    if common_count:
                        st.write(f"**Shared stocks found:** {common_count}")
                        overlap_bar = px.bar(common_df.head(10), x='Stock', y='OverlapWeight', title='Top Shared Stock Overlap Weights')
                        st.plotly_chart(overlap_bar, use_container_width=True)
                        overlap_table = common_df[['Stock', 'Weight_A', 'Weight_B', 'OverlapWeight']].rename(
                            columns={
                                'Weight_A': f'{fund_a_name} Weight',
                                'Weight_B': f'{fund_b_name} Weight',
                                'OverlapWeight': 'Overlap Weight'
                            }
                        )
                        st.dataframe(overlap_table, use_container_width=True)
                    else:
                        st.info("No common stock holdings were detected between the selected funds.")

                with nav_tab:
                    st.write("### NAV Comparison")
                    st.write("Compare the current Net Asset Value of both funds.")

                    # NAV Comparison
                    nav_col1, nav_col2 = st.columns(2)
                    with nav_col1:
                        st.metric(
                            label=f"{fund_a_name} NAV",
                            value=f"₹{fund_a_nav}" if fund_a_nav != 'N/A' else 'N/A',
                            delta=None
                        )
                    with nav_col2:
                        st.metric(
                            label=f"{fund_b_name} NAV",
                            value=f"₹{fund_b_nav}" if fund_b_nav != 'N/A' else 'N/A',
                            delta=None
                        )
                
            except Exception as e:
                if isinstance(e, json.JSONDecodeError):
                    st.error("AI returned invalid JSON. Please try again or verify the fund names.")
                else:
                    st.error("Error retrieving data. Please ensure the fund names are accurate and try again.")

                if DEBUG_MODE:
                    st.markdown("**Debug output:**")
                    st.code(clean_response or raw_response or "<no response available>")
                    st.write(str(e))
    else:
        st.warning("Please enter both fund names to run the audit.")

# ==========================================
# 5. FOOTER & LEGAL (Crucial for Compliance)
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
footer_col1, footer_col2 = st.columns([3,1])

with footer_col1:
    st.caption("""
    **© 2026 The Fund Audit** | Powered by Gemini AI.  
    **Disclaimer:** This tool is for educational and informational purposes only. The holdings data is AI-generated and may not reflect real-time AMC disclosures. This is NOT investment advice. We are not a SEBI-registered advisor. Please consult a Certified Financial Planner (CFP) before making any investment decisions.
    """)

with footer_col2:
    st.markdown("thefundaudit.mail@gmail.com")

# This part hides the Streamlit "Made with Streamlit" footer for a cleaner brand look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {
                visibility: hidden;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            section.main {
                padding-top: 0 !important;
            }
            div.block-container {
                padding-top: 0 !important;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)