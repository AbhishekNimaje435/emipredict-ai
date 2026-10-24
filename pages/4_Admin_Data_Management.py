import os
import sqlite3

import pandas as pd
import streamlit as st

from theme import apply_theme

from step3_feature_engineering import CAT_COLS
from utils import (DATA_DIR, INPUT_COLS, TARGET_CLF, TARGET_REG, load_categories, load_sample,
                   predict, render_inputs)

st.set_page_config(page_title="Admin - Data Management", page_icon="🛠️", layout="wide")
apply_theme()
st.title("🛠️ Financial Records Management (CRUD)")

# ------------------------------------------------------------------ auth
def admin_password() -> str:
    try:
        return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        return os.environ.get("ADMIN_PASSWORD", "admin123")


if not st.session_state.get("admin_ok"):
    pw = st.text_input("Admin password", type="password")
    if st.button("Login", type="primary"):
        if pw == admin_password():
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.caption("The demo default password is `admin123`. On deployment, set "
               "`ADMIN_PASSWORD` in Streamlit secrets.")
    st.stop()

# ------------------------------------------------------------------ db
DB_PATH = DATA_DIR / "emi_records.db"
ALL_COLS = INPUT_COLS + [TARGET_CLF, TARGET_REG]
TEXT_COLS = set(CAT_COLS + [TARGET_CLF])


def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db(force=False):
    if force and DB_PATH.exists():
        DB_PATH.unlink()
    if DB_PATH.exists():
        return
    cols = ", ".join(f"{c} {'TEXT' if c in TEXT_COLS else 'REAL'}" for c in ALL_COLS)
    with conn() as c:
        c.execute(f"CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        load_sample()[ALL_COLS].to_sql("records", c, if_exists="append", index=False)


init_db()
st.caption("Note: the filesystem on Streamlit Cloud is ephemeral - changes may reset to the "
           "sample data when the app restarts.")

with conn() as c:
    total = c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
st.metric("Total records", f"{total:,}")

cats = load_categories()
t_read, t_create, t_update, t_delete = st.tabs(["📖 Read", "➕ Create", "✏️ Update", "🗑️ Delete"])

# ------------------------------------------------------------------ read / filters
def build_query(scen, elig, rec_id):
    where, params = [], []
    if scen:
        where.append(f"emi_scenario IN ({','.join('?' * len(scen))})"); params += scen
    if elig:
        where.append(f"{TARGET_CLF} IN ({','.join('?' * len(elig))})"); params += elig
    if rec_id:
        where.append("id = ?"); params.append(int(rec_id))
    return (" WHERE " + " AND ".join(where)) if where else "", params


with t_read:
    f1, f2, f3, f4 = st.columns(4)
    scen = f1.multiselect("EMI scenario", cats["categorical"]["emi_scenario"], key="r_scen")
    elig = f2.multiselect("Eligibility", ["Eligible", "High_Risk", "Not_Eligible"], key="r_elig")
    rec_id = f3.number_input("Record id (0 = all)", min_value=0, value=0, step=1, key="r_id")
    page_size = f4.selectbox("Rows per page", [25, 50, 100], key="r_ps")

    where, params = build_query(scen, elig, rec_id)
    with conn() as c:
        n = c.execute(f"SELECT COUNT(*) FROM records{where}", params).fetchone()[0]
        pages = max((n - 1) // page_size + 1, 1)
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="r_page")
        view = pd.read_sql_query(
            f"SELECT * FROM records{where} ORDER BY id LIMIT ? OFFSET ?",
            c, params=params + [page_size, (page - 1) * page_size])
    st.caption(f"{n:,} records found - page {page}/{pages}")
    st.dataframe(view, hide_index=True)

# ------------------------------------------------------------------ create
with t_create:
    with st.form("create_record"):
        new = render_inputs("create")
        st.markdown("**Targets**")
        auto = st.checkbox("Auto-predict targets using the deployed models", value=True)
        g1, g2 = st.columns(2)
        man_clf = g1.selectbox("Eligibility (manual)", ["Eligible", "High_Risk", "Not_Eligible"])
        man_emi = g2.number_input("Max monthly EMI (manual)", min_value=0.0, value=10000.0, step=500.0)
        add = st.form_submit_button("Add record", type="primary")
    if add:
        if auto:
            res, _ = predict(pd.DataFrame([new]))
            new[TARGET_CLF] = res.loc[0, "predicted_eligibility"]
            new[TARGET_REG] = float(res.loc[0, "predicted_max_emi"])
        else:
            new[TARGET_CLF], new[TARGET_REG] = man_clf, man_emi
        with conn() as c:
            cur = c.execute(
                f"INSERT INTO records ({','.join(ALL_COLS)}) VALUES ({','.join('?' * len(ALL_COLS))})",
                [new[col] for col in ALL_COLS])
        st.success(f"Record added (id={cur.lastrowid}) - eligibility: {new[TARGET_CLF]}, "
                   f"max EMI: ₹{new[TARGET_REG]:,.0f}")

# ------------------------------------------------------------------ update
with t_update:
    st.write("Edit cells in the table below, then click **Save changes**. "
             "(The filters and page from the Read tab also apply here.)")
    edited = st.data_editor(view, hide_index=True, disabled=["id"], key=f"editor_{page}_{n}")
    if st.button("Save changes", type="primary"):
        changed = 0
        base = view.set_index("id")
        new_df = edited.set_index("id")
        with conn() as c:
            for rid in new_df.index:
                if not base.loc[rid].equals(new_df.loc[rid]):
                    vals = [None if pd.isna(new_df.loc[rid, col]) else new_df.loc[rid, col]
                            for col in ALL_COLS]
                    c.execute(f"UPDATE records SET {', '.join(f'{col}=?' for col in ALL_COLS)} "
                              "WHERE id=?", vals + [int(rid)])
                    changed += 1
        st.success(f"{changed} record(s) updated.")

# ------------------------------------------------------------------ delete
with t_delete:
    ids = st.multiselect("Record ids to delete (from the current page)", view["id"].tolist())
    confirm = st.checkbox("Yes, permanently delete these records")
    if st.button("Delete selected", type="primary", disabled=not (ids and confirm)):
        with conn() as c:
            c.executemany("DELETE FROM records WHERE id=?", [(int(i),) for i in ids])
        st.success(f"{len(ids)} record(s) deleted. Refresh the page.")
    st.divider()
    if st.button("Reset database to the original sample"):
        init_db(force=True)
        st.success("Database reset. Refresh the page.")
