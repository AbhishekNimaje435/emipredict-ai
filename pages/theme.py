"""Animated dark glass theme for EMIPredict AI. Call apply_theme() right after st.set_page_config()."""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
@property --n{syntax:'<integer>';initial-value:0;inherits:false}
:root{--teal:#5eead4;--violet:#8b7bff;--glass:rgba(255,255,255,.055);--line:rgba(255,255,255,.12)}
html,body,[class*="css"]{font-family:'Inter',sans-serif}
h1,h2,h3{font-family:'Poppins',sans-serif!important;letter-spacing:-.01em}

/* animated background */
.stApp{background:linear-gradient(-45deg,#070b1a,#0e1633,#0a1d33,#170f38);background-size:400% 400%;animation:bg 22s ease infinite}
.stApp::before{content:"";position:fixed;inset:-10%;z-index:0;pointer-events:none;
 background:radial-gradient(520px circle at 15% 20%,rgba(94,234,212,.20),transparent 60%),
 radial-gradient(640px circle at 85% 75%,rgba(139,123,255,.24),transparent 60%),
 radial-gradient(420px circle at 70% 10%,rgba(56,189,248,.14),transparent 60%);
 animation:orbs 18s ease-in-out infinite alternate}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent!important;position:relative;z-index:1}
[data-testid="stHeader"]{background:transparent!important}
#MainMenu,footer{visibility:hidden}
[data-testid="stSidebar"]{background:rgba(12,18,40,.72)!important;backdrop-filter:blur(16px);border-right:1px solid var(--line)}

/* entrance animations */
.block-container{animation:fadeUp .8s ease both}
[data-testid="stMetric"],[data-testid="stAlert"],[data-testid="stPlotlyChart"],[data-testid="stDataFrame"],
[data-testid="stForm"],[data-testid="stTabs"]{animation:fadeUp .7s ease both}
[data-testid="stAlert"]{animation-name:slideIn}

/* glass cards */
[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:16px 18px;
 backdrop-filter:blur(12px);transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease}
[data-testid="stMetric"]:hover{transform:translateY(-4px);border-color:rgba(94,234,212,.55);box-shadow:0 10px 30px rgba(94,234,212,.15)}
[data-testid="stMetricValue"]{font-family:'Poppins',sans-serif;background:linear-gradient(90deg,#fff,var(--teal));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
[data-testid="stForm"]{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:22px;backdrop-filter:blur(12px)}
[data-testid="stAlert"]{border-radius:14px;border:1px solid var(--line);backdrop-filter:blur(10px)}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line)}
[data-testid="stPlotlyChart"]{background:var(--glass);border:1px solid var(--line);border-radius:16px;padding:8px}
.js-plotly-plot .plotly .main-svg{background:transparent!important}

/* headings and inputs */
h1{background:linear-gradient(90deg,#fff 20%,var(--teal) 55%,var(--violet) 90%);background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 6s linear infinite}
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="base-input"]{border-radius:10px!important}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{box-shadow:0 0 0 2px rgba(94,234,212,.45)!important}
button[data-baseweb="tab"]{border-radius:999px;padding:6px 18px;transition:background .2s}
button[data-baseweb="tab"]:hover{background:rgba(94,234,212,.10)}
[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--teal),var(--violet))!important;height:3px!important;border-radius:3px}

/* buttons */
.stButton>button,[data-testid="stFormSubmitButton"]>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--line);font-weight:600;transition:all .25s ease}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 8px 24px rgba(94,234,212,.25);border-color:var(--teal)}
[data-testid="stFormSubmitButton"]>button[kind="primary"],.stButton>button[kind="primary"]{background:linear-gradient(90deg,#14b8a6,#6d5dfc);color:#fff;border:0}

/* page links as cards */
[data-testid="stPageLink"] a{background:var(--glass);border:1px solid var(--line);border-radius:14px;padding:16px 18px;backdrop-filter:blur(10px);transition:all .25s ease}
[data-testid="stPageLink"] a:hover{transform:translateY(-4px);border-color:var(--teal);box-shadow:0 12px 30px rgba(109,93,252,.25)}

/* hero */
.hero{text-align:center;padding:34px 10px 10px}
.hero>*{animation:fadeUp .9s ease both}
.hero .pill{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border:1px solid var(--line);border-radius:999px;background:var(--glass);font-size:.85rem}
.hero .dot{width:8px;height:8px;border-radius:50%;background:#34d399;animation:pulse 1.6s infinite}
.hero-title{font:700 clamp(2.6rem,7vw,4.6rem)/1.05 'Poppins',sans-serif!important;margin:.35em 0 .1em;animation-delay:.1s}
.hero-title span{background:linear-gradient(90deg,var(--teal),var(--violet),var(--teal));background-size:200% auto;-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 4s linear infinite}
.hero-sub{font:600 1.35rem 'Poppins',sans-serif;margin:0;animation-delay:.25s}
.hero-desc{max-width:640px;margin:.8rem auto 0;color:#aeb8d4;animation-delay:.4s}
.stats{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin:28px 0 8px;animation-delay:.55s}
.stat{min-width:170px;padding:16px 22px;background:var(--glass);border:1px solid var(--line);border-radius:16px;backdrop-filter:blur(12px);animation:float 5s ease-in-out infinite}
.stat:nth-child(2){animation-delay:.6s}.stat:nth-child(3){animation-delay:1.2s}.stat:nth-child(4){animation-delay:1.8s}
.stat b{display:block;font:700 2rem 'Poppins',sans-serif;color:var(--teal)}
.stat span{font-size:.85rem;color:#aeb8d4}
.cnt{counter-reset:n var(--n)}
.cnt::after{content:counter(n)}
.c400{animation:k400 2.4s cubic-bezier(.2,.7,.2,1) .6s both}.c400::after{content:counter(n) "K+"}
.c97{animation:k97 2.4s cubic-bezier(.2,.7,.2,1) .6s both}.c97::after{content:counter(n) ".5%"}
.c746{animation:k746 2.4s cubic-bezier(.2,.7,.2,1) .6s both}.c746::after{content:"\20B9" counter(n)}
.c8{animation:k8 1.6s cubic-bezier(.2,.7,.2,1) .6s both}
@keyframes k400{to{--n:400}}@keyframes k97{to{--n:97}}@keyframes k746{to{--n:746}}@keyframes k8{to{--n:8}}

@keyframes bg{0%{background-position:0 50%}50%{background-position:100% 50%}100%{background-position:0 50%}}
@keyframes orbs{from{transform:translate3d(0,0,0) scale(1)}to{transform:translate3d(-3%,4%,0) scale(1.12)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
@keyframes slideIn{from{opacity:0;transform:translateX(-26px)}to{opacity:1;transform:none}}
@keyframes shine{to{background-position:200% center}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.6)}100%{box-shadow:0 0 0 10px rgba(52,211,153,0)}}
::-webkit-scrollbar{width:10px}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.18);border-radius:8px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}
 .c400{--n:400}.c97{--n:97}.c746{--n:746}.c8{--n:8}}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)
