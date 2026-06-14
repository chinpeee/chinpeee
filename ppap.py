#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# =====================================================
# 中文字型
# =====================================================

from matplotlib import font_manager

font_path = "NotoSansTC-Regular.ttf"

font_manager.fontManager.addfont(font_path)

plt.rcParams['font.family'] = 'Noto Sans TC'
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 載入 PROCESS 資料
# =====================================================

process_df = pd.read_excel("028482020.xlsx")

# =====================================================
# 原本的縣市資料
# =====================================================

national_avg = 1.5016

data = {
    "縣市": [
        "基隆市", "台北市", "新北市", "桃園市", "新竹縣", "新竹市",
        "苗栗縣", "南投縣", "台中市", "彰化縣", "雲林縣", "嘉義縣",
        "嘉義市", "台南市", "高雄市", "屏東縣", "宜蘭縣", "花蓮縣",
        "台東縣", "澎湖縣", "金門縣", "連江縣"
    ],

    "平均媒介管道數": [
        1.5144,1.5598,1.6357,1.5213,1.4424,1.5673,
        1.4239,1.3963,1.4834,1.4854,1.4944,1.3499,
        1.4423,1.4063,1.5130,1.5319,1.5000,1.3958,
        1.4545,1.2857,1.0000,1.6000
    ],

    "lat":[
        25.1082,25.0330,24.9664,24.9376,24.7001,
        24.8143,24.5649,23.8388,24.2332,23.9912,
        23.7092,23.4518,23.4755,23.1417,22.7265,
        22.4701,24.6022,23.7569,22.7972,23.5654,
        24.4482,26.1521
    ],

    "lon":[
        121.7081,121.5654,121.4367,121.2168,121.1235,
        120.9675,120.8208,120.9876,120.8170,120.4811,
        120.4313,120.5746,120.4473,120.2513,120.3954,
        120.5891,121.6291,121.3542,121.0711,119.5664,
        118.3786,119.9281
    ]
}

df = pd.DataFrame(data)

df["與全國平均差距"] = (
    df["平均媒介管道數"] - national_avg
)

min_val = df["平均媒介管道數"].min()
max_val = df["平均媒介管道數"].max()

df["map_size"] = (
    ((df["平均媒介管道數"] - min_val) /
    (max_val - min_val)) ** 3
) * 25000 + 300

# =====================================================
# Streamlit UI
# =====================================================

st.title("📊 媒介資訊不平等 Dashboard")

st.sidebar.header("控制面板")

chart_type = st.sidebar.radio(
    "趨勢圖表類型",
    ["長條圖 (Bar)", "折線圖 (Line)"]
)

sort_order = st.sidebar.selectbox(
    "資料排序方式",
    ["依地理預設", "數值由高到低", "數值由低到高"]
)

if sort_order == "數值由高到低":
    df_display = df.sort_values(
        by="平均媒介管道數",
        ascending=False
    )

elif sort_order == "數值由低到高":
    df_display = df.sort_values(
        by="平均媒介管道數",
        ascending=True
    )

else:
    df_display = df

# =====================================================
# PROCESS MODEL 7
# =====================================================

st.header("🧠 PROCESS Model 7 路徑圖")

fig, ax = plt.subplots(figsize=(10,5))

ax.axis("off")

ax.text(
    0.1,0.5,
    "資訊近用\n(X)",
    fontsize=16,
    bbox=dict(boxstyle="round")
)

ax.text(
    0.45,0.5,
    "風險感知\n(M)",
    fontsize=16,
    bbox=dict(boxstyle="round")
)

ax.text(
    0.8,0.5,
    "預防行為\n(Y)",
    fontsize=16,
    bbox=dict(boxstyle="round")
)

ax.text(
    0.45,0.15,
    "媒介類型\n(W)",
    fontsize=16,
    bbox=dict(boxstyle="round")
)

ax.arrow(0.2,0.55,0.18,0,head_width=0.02)
ax.arrow(0.55,0.55,0.18,0,head_width=0.02)
ax.arrow(0.48,0.25,0,0.18,head_width=0.02)

st.pyplot(fig)

# =====================================================
# X → M
# =====================================================

st.header("📈 資訊近用對風險感知的影響 X → M")

fig, ax = plt.subplots(figsize=(8,5))

sns.regplot(
    data=process_df,
    x="info",
    y="risk",
    scatter_kws={"alpha":0.3},
    ax=ax
)

ax.set_title("資訊近用對風險感知")
ax.set_xlabel("資訊近用")
ax.set_ylabel("風險感知")

st.pyplot(fig)

# =====================================================
# M → Y
# =====================================================

st.header("📈 風險感知對預防行為的影響 M → Y")

fig, ax = plt.subplots(figsize=(8,5))

sns.regplot(
    data=process_df,
    x="risk",
    y="action",
    scatter_kws={"alpha":0.3},
    ax=ax
)

ax.set_title("風險感知對預防行為")
ax.set_xlabel("風險感知")
ax.set_ylabel("預防行為")

st.pyplot(fig)

# =====================================================
# CORRELATION
# =====================================================

st.header("🔥 主要變項相關矩陣")

corr = process_df[
    ["info","media","risk","action"]
].corr()

fig, ax = plt.subplots(figsize=(6,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="RdYlBu_r",
    fmt=".2f",
    ax=ax
)

ax.set_title("主要變項相關矩陣")

st.pyplot(fig)

# =====================================================
# SIMPLE SLOPE
# =====================================================

st.header("📊 媒介類型的調節效果（Simple Slope）")

process_df["interaction"] = (
    process_df["info"] *
    process_df["media"]
)

X = process_df[
    ["info","media","interaction"]
]

X = sm.add_constant(X)

y = process_df["risk"]

model = sm.OLS(y,X).fit()

b0 = model.params["const"]
b1 = model.params["info"]
b2 = model.params["media"]
b3 = model.params["interaction"]

media_mean = process_df["media"].mean()
media_sd = process_df["media"].std()

low_media = media_mean - media_sd
high_media = media_mean + media_sd

info_range = np.linspace(
    process_df["info"].min(),
    process_df["info"].max(),
    100
)

risk_low = (
    b0 +
    b1*info_range +
    b2*low_media +
    b3*(info_range*low_media)
)

risk_high = (
    b0 +
    b1*info_range +
    b2*high_media +
    b3*(info_range*high_media)
)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    info_range,
    risk_low,
    linewidth=3,
    label="低媒介類型"
)

ax.plot(
    info_range,
    risk_high,
    linewidth=3,
    label="高媒介類型"
)

ax.set_title("媒介類型對資訊近用與風險感知關係之調節效果")

ax.set_xlabel("資訊近用")
ax.set_ylabel("風險感知")

ax.legend()

st.pyplot(fig)

# =====================================================
# 原有儀表板內容
# =====================================================

st.markdown("---")

st.header("📌 全國與地方趨勢")

st.metric(
    label="全國平均媒介管道數值",
    value=national_avg
)

chart_data = df_display.set_index("縣市")[["平均媒介管道數"]]

if chart_type == "長條圖 (Bar)":
    st.bar_chart(chart_data)
else:
    st.line_chart(chart_data)

# =====================================================
# 離差圖
# =====================================================

st.markdown("---")

st.header("📉 各縣市與全國平均差距")

deviation_data = df_display.set_index("縣市")[["與全國平均差距"]]

st.bar_chart(deviation_data)

# =====================================================
# 地圖
# =====================================================

st.markdown("---")

st.header("🌐 全台空間分佈")

st.map(
    df,
    latitude="lat",
    longitude="lon",
    size="map_size"
)

# =====================================================
# 資料表
# =====================================================

st.markdown("---")

st.header("📋 原始數據與表格內視覺化")

st.dataframe(
    df_display[
        ["縣市","平均媒介管道數","與全國平均差距"]
    ],
    column_config={
        "平均媒介管道數":
        st.column_config.ProgressColumn(
            "平均媒介管道數",
            format="%.4f",
            min_value=0,
            max_value=2.0
        ),

        "與全國平均差距":
        st.column_config.NumberColumn(
            "與全國平均差距",
            format="%.4f"
        )
    },
    hide_index=True,
    use_container_width=True
)

