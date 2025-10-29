import streamlit as st
import pandas as pd
import altair as alt 
import os
import numpy as np
from pathlib import Path

# --- 1. 頁面設定 (Page Config) ---
st.set_page_config(
    page_title="張信哲 (Jeff Chang) 歌詞與音樂分析",
    page_icon="🎵",
    layout="wide",
)

# --- 2. 檔案路徑設定 ---
DATA_FILE_NAME = 'Jeff_Chang_Final_Master_Dashboard_Data.csv'
AI_COLS = ['ai_theme', 'ai_sentiment', 'ai_notes']

# --- 3. 輔助函式 (Helper Functions) ---

@st.cache_data(show_spinner=True) # 載入時顯示轉圈圈，提高使用者體驗
def load_data_from_disk():
    """ 載入單一的主儀表板資料檔案。 """
    
    DATA_FILE = Path(__file__).parent / DATA_FILE_NAME
    
    if not DATA_FILE.exists():
        st.error(f"致命錯誤：找不到儀表板主資料檔案: {DATA_FILE_NAME}")
        st.stop()
        return None
    
    try:
        df = pd.read_csv(str(DATA_FILE), encoding='utf-8-sig', low_memory=False)
        return df
    except Exception as e:
        st.error(f"載入 {DATA_FILE_NAME} 時發生嚴重錯誤，請檢查檔案內容和編碼: {e}")
        return None

def initialize_session_state(df):
    """ 初始化 session_state，只運行一次。 """
    if 'initialized' not in st.session_state:
        # 檢查 AI 欄位是否存在
        if all(col in df.columns for col in AI_COLS):
            df['has_ai_analysis'] = df['ai_theme'].notna() & (~df['ai_theme'].isin(['SKIPPED', 'ERROR']))
            st.session_state['ai_available'] = True
        else:
            df['has_ai_analysis'] = False
            st.session_state['ai_available'] = False
            st.warning("AI 分析資料欄位不存在。部分功能將受影響。")
        
        st.session_state['initialized'] = True
    
    # 確保在每次運行時，我們都使用計算好的 'has_ai_analysis' 欄位
    # 如果 session_state 已經初始化，df 會帶有這個欄位
    if 'has_ai_analysis' not in df.columns:
        df['has_ai_analysis'] = False # 確保它不會導致後續代碼崩潰
    
    return df


# 將 load_data_from_disk 的結果傳遞給 initialize_session_state
@st.cache_data(show_spinner=True)
def get_final_data():
    df = load_data_from_disk()
    if df is not None:
        df = initialize_session_state(df)
    return df
    

@st.cache_data
def plot_categorical_chart(df, column, title, top_n=15):
    """ 繪製分類型別的長條圖 """
    if column not in df.columns or df[column].dropna().empty: return None
    data = df.dropna(subset=[column])
    
    if column == 'key_key':
        key_map = {
            0.0: 'C', 1.0: 'C#', 2.0: 'D', 3.0: 'D#', 4.0: 'E', 5.0: 'F',
            6.0: 'F#', 7.0: 'G', 8.0: 'G#', 9.0: 'A', 10.0: 'A#', 11.0: 'B'
        }
        data['key_name'] = data[column].apply(lambda x: key_map.get(x, pd.NA))
        data = data.dropna(subset=['key_name'])
        column = 'key_name'
        title = "歌曲調性 (Key)"

    chart_data = data[column].value_counts().head(top_n).reset_index()
    chart_data.columns = [column, 'count']

    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X(column, title=title, sort='-y'),
        y=alt.Y('count', title='歌曲數量 (Count)'),
        color=alt.Color(column, title=title, legend=None),
        tooltip=[column, 'count']
    ).properties(
        title=f"{title} 分佈 (Top {top_n})" if column != 'key_name' else f"{title} 分佈"
    ).interactive()
    return chart

@st.cache_data
def plot_histogram(df, column, title, bin_count=10):
    """ 繪製數值型別的直方圖 (Histogram) """
    if column not in df.columns or df[column].dropna().empty: return None
    data = df.dropna(subset=[column])

    chart = alt.Chart(data).mark_bar().encode(
        alt.X(column, bin=alt.Bin(maxbins=bin_count), title=title),
        alt.Y('count()', title='歌曲數量 (Count)'),
        tooltip=[alt.Tooltip(column, bin=True), 'count()']
    ).properties(
        title=f"{title} 分佈 (直方圖)"
    ).interactive()
    return chart

# --- 4. 儀表板主要應用程式 ---
def main():
    
    # 透過更穩健的快取和初始化機制獲取數據
    df = get_final_data()

    # --- 4a. 處理資料載入失敗 ---
    if df is None:
        st.title("張信哲 (Jeff Chang) AI 歌詞分析儀表板")
        st.error("資料載入失敗，請檢查 Streamlit 應用程式日誌。")
        return

    # --- 4b. 成功的資料載入 ---
    
    # --- 側邊欄導航 (Sidebar) ---
    st.sidebar.title("導航 (Navigation)")
    st.sidebar.markdown("從下方選擇一首歌曲以查看詳細分析。若不選擇，將顯示主儀表板。")
    
    if 'track_name' not in df.columns: df['track_name'] = 'N/A' 
    if 'album_title' not in df.columns: df['album_title'] = 'N/A'
        
    df['display_name'] = df['track_name'].fillna('N/A') + " | " + df['album_title'].fillna('N/A')
    
    # 排序邏輯：AI 分析在前，然後按名稱排序
    df_sorted_for_list = df.sort_values(
        by=['has_ai_analysis', 'display_name'],
        ascending=[False, True]
    )
    sorted_unique_names = df_sorted_for_list['display_name'].unique().tolist()
    
    song_list = ['[ 主儀表板 (General Dashboard) ]'] + sorted_unique_names
    
    selected_song = st.sidebar.selectbox(
        "選擇一首歌曲 (Select a Song)",
        options=song_list,
        index=0  
    )

    # --- 5. 頁面邏輯 ---

    if selected_song == '[ 主儀表板 (General Dashboard) ]':
        st.title("張信哲 (Jeff Chang) AI 歌詞分析儀表板 v1.7 [Final]") # 更新版本號
        
        # 統計數據
        total_songs = len(df)
        songs_with_lyrics = df['lyrics_text'].notna().sum()
        songs_with_ai = (df['has_ai_analysis'] == True).sum()

        st.info(f"總歌曲數: {total_songs} | 包含歌詞: {songs_with_lyrics} 筆 | 已獲 AI 分析: {songs_with_ai} 筆")
        
        st.header("總體分析 (Overall Analysis)")
        
        col1, col2 = st.columns(2)
        
        # 只有在 AI 可用時才顯示圖表
        if st.session_state.get('ai_available', False) and songs_with_ai > 0:
            df_analyzed = df[df['has_ai_analysis'] == True]
            
            if not df_analyzed.empty:
                with col1:
                    # 圖表 1: AI 分析的情緒分佈
                    st.subheader("AI 分析的情緒分佈")
                    sentiment_counts = df_analyzed['ai_sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['情緒 (Sentiment)', '歌曲數量 (Count)']
                    
                    chart_sentiment = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta("歌曲數量 (Count)", stack=True),
                        color=alt.Color("情緒 (Sentiment)"),
                        tooltip=["情緒 (Sentiment)", "歌曲數量 (Count)"]
                    ).properties(title="AI 分析的情緒")
                    st.altair_chart(chart_sentiment, use_container_width=True)

                with col2:
                    # 圖表 2: AI 分析的主題分佈
                    st.subheader("AI 分析的主題分佈")
                    theme_counts = df_analyzed['ai_theme'].value_counts().head(10).reset_index()
                    theme_counts.columns = ['主題 (Theme)', '歌曲數量 (Count)']
                    
                    chart_theme = alt.Chart(theme_counts).mark_bar().encode(
                        x=alt.X("主題 (Theme)", sort='-y'),
                        y=alt.Y("歌曲數量 (Count)"),
                        color="主題 (Theme)",
                        tooltip=["主題 (Theme)", "歌曲數量 (Count)"]
                    ).properties(title="前 10 大 AI 分析主題")
                    st.altair_chart(chart_theme, use_container_width=True)
            else:
                st.warning("AI 分析資料已載入，但似乎不包含有效的情緒或主題資料。")
        else:
            st.warning("AI 分析資料未找到或無有效分析結果。圖表無法顯示。")
            st.write("請運行 AI 分析腳本並將資料合併到單一的 CSV 檔案中。")

        # === 音訊資料維度分析 ===
        st.divider() 
        st.header("音訊資料維度分析 (Tonal Data Dimensions)")

        st.subheader("分類型資料 (Categorical Data)")
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            if chart_genre := plot_categorical_chart(df, 'genre_ros', '音樂流派 (Genre)', top_n=15):
                st.altair_chart(chart_genre, use_container_width=True)
        
        with chart_col2:
            if chart_scale := plot_categorical_chart(df, 'key_scale', '音樂調式 (大/小調)'):
                st.altair_chart(chart_scale, use_container_width=True)
        
        with chart_col3:
            if chart_key := plot_categorical_chart(df, 'key_key', '歌曲調性 (Key) 分佈', top_n=12):
                st.altair_chart(chart_key, use_container_width=True)

        st.subheader("數值型資料 (Numerical Data)")
        chart_col4, chart_col5 = st.columns(2)

        with chart_col4:
            if chart_party := plot_histogram(df, 'mood_party', '派對指數 (Mood: Party)'):
                st.altair_chart(chart_party, use_container_width=True)
            
        with chart_col5:
            if chart_dance := plot_histogram(df, 'danceability', '舞蹈指數 (Danceability)'):
                st.altair_chart(chart_dance, use_container_width=True)
        
    # --- 5b. 單曲分析頁面 ---
    else:
        song_data = df[df['display_name'] == selected_song].iloc[0]
        
        st.title(f"🎵 {song_data['track_name']}")
        st.subheader(f"專輯 (Album): *{song_data.get('album_title', 'N/A')}*")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1]) 
        
        with col1:
            # 歌詞與 AI 分析
            st.header("歌詞與分析")
            
            # 顯示歌詞
            st.markdown("### 歌詞 (Lyrics)")
            if pd.notna(song_data.get('lyrics_text')):
                st.text_area("Lyrics", song_data['lyrics_text'], height=300, label_visibility="collapsed")
            else:
                st.info("此歌曲無歌詞資料。")
            
            # 顯示 AI 分析 (如果存在)
            st.markdown("### AI 綜合分析 (AI Analysis)")
            if st.session_state.get('ai_available', False) and pd.notna(song_data.get('ai_theme')) and song_data.get('ai_theme') not in ['SKIPPED', 'ERROR']:
                st.info(f"**AI 主題 (Theme):**\n{song_data['ai_theme']}")
                st.warning(f"**AI 情緒 (Sentiment):**\n{song_data['ai_sentiment']}")
                st.markdown("**AI 綜合筆記 (Notes):**")
                st.write(song_data['ai_notes'])
            else:
                st.info("此歌曲尚無 AI 分析資料。")
                
            # 顯示製作人員
            st.markdown("---")
            st.markdown("#### 製作人員 (Credits)")
            credits_cols = st.columns(2)
            credits_cols[0].markdown(f"**作詞:** {song_data.get('作詞', 'N/A')}")
            credits_cols[1].markdown(f"**作曲:** {song_data.get('作曲', 'N/A')}")
            credits_cols[0].markdown(f"**製作:** {song_data.get('製作', 'N/A')}")
            credits_cols[1].markdown(f"**編曲:** {song_data.get('編曲', 'N/A')}")

        with col2:
            # 所有其他資料欄位
            st.header("所有資料欄位 (All Data Fields)")
            manual_cols = [
                'track_name', 'album_title', 'lyrics_text', '作詞', '作曲', '製作', '編曲',
                'ai_theme', 'ai_sentiment', 'ai_notes', 'display_name', 'has_ai_analysis',
            ]
            
            other_fields = song_data.drop(labels=manual_cols, errors='ignore')
            other_fields_with_data = other_fields.dropna()
            
            if not other_fields_with_data.empty:
                st.dataframe(other_fields_with_data, use_container_width=True)
            else:
                st.info("此歌曲沒有其他可用的 (Tonal/AcousticBrainz) 資料。")

# --- 6. 執行 Main ---
if __name__ == "__main__":
    # 確保 session_state 在應用程式開始時只初始化一次
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False
        st.session_state['ai_available'] = False
    main()

