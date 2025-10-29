import streamlit as st
import pandas as pd
import altair as alt 
import numpy as np
from pathlib import Path
import os

# --- 1. 頁面設定 (Page Config) ---
st.set_page_config(
    page_title="張信哲 (Jeff Chang) 歌詞與音樂分析",
    page_icon="🎵",
    layout="wide",
)

# --- 2. 檔案路徑設定 ---
DATA_FILE_NAME = 'Jeff_Chang_Final_Master_Dashboard_Data.csv'
AI_COLS_CHECK = ['ai_theme', 'ai_sentiment', 'ai_notes'] # 用於檢查 AI 數據是否存在的欄位
MOOD_COLS = ['mood_aggressive', 'mood_happy', 'mood_party', 'mood_relaxed', 'mood_sad'] # 明確定義 Mood 欄位

# --- 3. 輔助函式 (Helper Functions) ---

@st.cache_data(show_spinner="正在載入並清洗資料...", persist=True) # 更新提示
def load_data_from_disk():
    """ 載入單一的主儀表板資料檔案，並在載入時強制清洗 Mood 欄位。如果失敗返回 None。 """
    
    DATA_FILE = Path(__file__).parent / DATA_FILE_NAME
    
    if not DATA_FILE.exists():
        return None # 讓 main() 處理錯誤顯示
    
    try:
        df = pd.read_csv(str(DATA_FILE), encoding='utf-8-sig', low_memory=False)
        
        # *** 關鍵修復：在快取前強制清洗 Mood 欄位 ***
        print("Attempting to clean mood columns...") # 添加日誌
        mood_cols_present = [col for col in MOOD_COLS if col in df.columns]
        for col in mood_cols_present:
            original_type = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_type = df[col].dtype
            nan_count = df[col].isna().sum()
            print(f"Cleaned '{col}': Original type {original_type}, New type {new_type}, NaNs after coerce: {nan_count}")
            
        # 為了安全起見，移除轉換後完全變成 NaN 的欄位 (雖然不太可能)
        # df.dropna(axis=1, how='all', inplace=True) 
            
        return df
    except Exception as e:
        # st.error(f"載入或清洗 {DATA_FILE_NAME} 時發生嚴重錯誤: {e}") # 在 main 中顯示錯誤
        print(f"Error during load or cleaning: {e}") # 打印到日誌
        return None

# (initialize_data_and_state 和 get_final_data 被移除，邏輯移入 main)

@st.cache_data
def plot_categorical_chart(df, column, title, top_n=15):
    """ 繪製分類型別的長條圖 """
    if column not in df.columns or df[column].dropna().empty: return None
    data = df.dropna(subset=[column]).copy()
    
    if column == 'key_key':
        key_map = {
            0.0: 'C', 1.0: 'C#', 2.0: 'D', 3.0: 'D#', 4.0: 'E', 5.0: 'F',
            6.0: 'F#', 7.0: 'G', 8.0: 'G#', 9.0: 'A', 10.0: 'A#', 11.0: 'B'
        }
        data.loc[:, 'key_name'] = data[column].apply(lambda x: key_map.get(x, pd.NA))
        data = data.dropna(subset=['key_name'])
        if data.empty: return None 
        column = 'key_name'
        title = "歌曲調性 (Key)"

    if data.empty: return None
    chart_data = data[column].value_counts().head(top_n).reset_index()
    if chart_data.empty: return None 
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
    # 數據在載入時已被清洗為數值型別
    data = df.dropna(subset=[column]).copy()
    if data.empty: return None 

    chart = alt.Chart(data).mark_bar().encode(
        alt.X(column, bin=alt.Bin(maxbins=bin_count), title=title),
        alt.Y('count()', title='歌曲數量 (Count)'),
        tooltip=[alt.Tooltip(column, bin=True), 'count()']
    ).properties(
        title=f"{title} 分佈 (直方圖)"
    ).interactive()
    return chart

@st.cache_data
def plot_pie_chart(df, column, title, top_n=10):
    """ 繪製分類型別的餅圖 """
    if column not in df.columns or df[column].dropna().empty: return None
    data = df.dropna(subset=[column]).copy()
    if data.empty: return None

    chart_data = data[column].value_counts().head(top_n).reset_index()
    if chart_data.empty: return None
    chart_data.columns = [column, 'count']
    
    total = chart_data['count'].sum()
    chart_data['percent'] = (chart_data['count'] / total)

    base = alt.Chart(chart_data).encode(theta=alt.Theta("count", stack=True))
    
    pie = base.mark_arc(outerRadius=120, innerRadius=50).encode(
        color=alt.Color(column, title=title),
        order=alt.Order("count", sort="descending"),
        tooltip=[column, 'count', alt.Tooltip('percent', format='.1%')]
    )
    
    text = base.mark_text(radius=140).encode(
        text=alt.Text(column, title=title),
        order=alt.Order("count", sort="descending"),
        color=alt.value("black")
    )

    chart = (pie + text).properties(title=f"{title} 分佈 (Top {top_n})")
    return chart

# --- 4. 儀表板主要應用程式 ---
def main():
    
    # *** 關鍵修復：每次運行都從快取載入數據 ***
    df_raw = load_data_from_disk() # 數據在載入時已被清洗

    # --- 4a. 處理資料載入失敗 ---
    if df_raw is None:
        st.title("張信哲 (Jeff Chang) AI 歌詞分析儀表板")
        st.error(f"致命錯誤：無法載入主資料檔案 '{DATA_FILE_NAME}'。請檢查檔案是否存在於 GitHub 儲存庫並重新部署。")
        st.stop() # 停止執行
        
    # 創建一個工作副本
    df = df_raw.copy()

    # *** 關鍵修復：將初始化邏輯移至 main() 內部 ***
    ai_available = False
    if all(col in df.columns for col in AI_COLS_CHECK):
        df['has_ai_analysis'] = df['ai_theme'].notna() & (~df['ai_theme'].isin(['SKIPPED', 'ERROR']))
        ai_available = True 
    else:
        df['has_ai_analysis'] = False 

    # --- 4b. 成功的資料載入 ---
    
    # --- 側邊欄導航 (Sidebar) ---
    st.sidebar.title("導航 (Navigation)")
    st.sidebar.markdown("從下方選擇一首歌曲以查看詳細分析。若不選擇，將顯示主儀表板。")
    
    # 確保 track_name 和 album_title 存在
    if 'track_name' not in df.columns: df['track_name'] = 'N/A' 
    if 'album_title' not in df.columns: df['album_title'] = 'N/A'
        
    df['display_name'] = df['track_name'].fillna('N/A') + " | " + df['album_title'].fillna('N/A')
    
    # 排序邏輯：AI 分析在前，然後按名稱排序
    if 'has_ai_analysis' in df.columns:
        df_sorted_for_list = df.sort_values(
            by=['has_ai_analysis', 'display_name'],
            ascending=[False, True]
        )
    else: 
        df_sorted_for_list = df.sort_values(by='display_name', ascending=True)

    sorted_unique_names = df_sorted_for_list['display_name'].unique().tolist()
    
    song_list = ['[ 主儀表板 (General Dashboard) ]'] + sorted_unique_names
    
    selected_song = st.sidebar.selectbox(
        "選擇一首歌曲 (Select a Song)",
        options=song_list,
        index=0  
    )

    # --- 5. 頁面邏輯 ---

    if selected_song == '[ 主儀表板 (General Dashboard) ]':
        st.title("張信哲 (Jeff Chang) AI 歌詞分析儀表板 v1.18 [最終穩定版]") # 更新版本號
        
        # 統計數據
        total_songs = len(df)
        songs_with_lyrics = 0
        if 'lyrics_text' in df.columns:
            songs_with_lyrics = df['lyrics_text'].notna().sum()
            
        # 直接從 DataFrame 計算
        songs_with_ai = (df['has_ai_analysis'] == True).sum() if 'has_ai_analysis' in df.columns else 0

        st.info(f"總歌曲數: {total_songs} | 包含歌詞: {songs_with_lyrics} 筆 | 已獲 AI 分析: {songs_with_ai} 筆")
        
        st.header("總體分析 (Overall Analysis)")
        
        col1, col2 = st.columns(2)
        
        # 使用本地變數 ai_available
        if ai_available and songs_with_ai > 0:
            df_analyzed = df[df['has_ai_analysis'] == True].copy() # 再次確保是副本
            
            # 再次檢查 df_analyzed 是否為空
            if not df_analyzed.empty:
                with col1:
                    st.subheader("AI 分析的情緒類別分佈")
                    if 'ai_sentiment_category' in df_analyzed.columns:
                         sentiment_counts = df_analyzed['ai_sentiment_category'].value_counts().reset_index()
                         sentiment_counts.columns = ['情緒類別 (Category)', '歌曲數量 (Count)']
                         
                         chart_sentiment = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                             theta=alt.Theta("歌曲數量 (Count)", stack=True),
                             color=alt.Color("情緒類別 (Category)"),
                             tooltip=["情緒類別 (Category)", "歌曲數量 (Count)"]
                         ).properties(title="AI 分析的情緒類別")
                         st.altair_chart(chart_sentiment, use_container_width=True)
                    else:
                         st.caption("欄位 'ai_sentiment_category' 不存在。")


                with col2:
                    st.subheader("AI 分析的主題分佈")
                    if 'ai_theme' in df_analyzed.columns:
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
                        st.caption("欄位 'ai_theme' 不存在。")
            else:
                st.warning("AI 分析資料已載入，但似乎不包含有效的情緒或主題資料。")
        elif not ai_available:
            st.warning("AI 分析資料欄位未找到。圖表無法顯示。")
        else: # ai_available is True but songs_with_ai is 0
             st.warning("AI 分析資料欄位存在，但沒有找到有效的 AI 分析結果。圖表無法顯示。")


        # === 音訊資料維度分析 ===
        st.divider() 
        st.header("音訊資料維度分析 (Tonal Data Dimensions)")

        st.subheader("分類型資料 (Categorical Data)")
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
             if chart_genre := plot_categorical_chart(df, 'genre_ros', '音樂流派 (Genre)', top_n=15):
                 st.altair_chart(chart_genre, use_container_width=True)
        
        with chart_col2:
            # 合併 key 和 scale
            df_key_scale = df.dropna(subset=['key_key', 'key_scale']).copy()
            if not df_key_scale.empty:
                key_map = {
                    0.0: 'C', 1.0: 'C#', 2.0: 'D', 3.0: 'D#', 4.0: 'E', 5.0: 'F',
                    6.0: 'F#', 7.0: 'G', 8.0: 'G#', 9.0: 'A', 10.0: 'A#', 11.0: 'B'
                }
                # 使用 .loc 避免警告
                df_key_scale.loc[:, 'key_name'] = df_key_scale['key_key'].apply(lambda x: key_map.get(x, 'N/A'))
                df_key_scale.loc[:, 'key_scale_combined'] = df_key_scale['key_name'] + ' ' + df_key_scale['key_scale']
                
                if chart_ks := plot_pie_chart(df_key_scale, 'key_scale_combined', '調性與調式組合'):
                    st.altair_chart(chart_ks, use_container_width=True)

        with chart_col3:
             if chart_timbre := plot_pie_chart(df, 'timbre', '音色 (Timbre)'):
                 st.altair_chart(chart_timbre, use_container_width=True)


        st.subheader("數值型資料 (Numerical Data)")
        # Mood 圖表
        mood_cols_present_in_df = [col for col in MOOD_COLS if col in df.columns]
        if mood_cols_present_in_df:
             # 動態決定欄數，最多 5 欄
            num_mood_cols = len(mood_cols_present_in_df)
            cols_per_row = min(num_mood_cols, 5) 
            mood_chart_cols = st.columns(cols_per_row)
            
            for i, mood_col in enumerate(mood_cols_present_in_df):
                with mood_chart_cols[i % cols_per_row]: # 使用取模運算符來循環使用欄位
                    # 數據已在 load_data_from_disk 中清洗
                    df_mood = df.dropna(subset=[mood_col]).copy() # 只需 dropna
                    
                    if not df_mood.empty: 
                        # 現在可以安全地進行比較 (因為已清洗)
                        df_mood.loc[:, f'{mood_col}_bin'] = np.where(df_mood[mood_col] >= 0.5, '高 (>=0.5)', '低 (<0.5)')
                        mood_title = mood_col.replace('mood_', '').capitalize()
                        if chart_mood_pie := plot_pie_chart(df_mood, f'{mood_col}_bin', f'Mood: {mood_title}'):
                            st.altair_chart(chart_mood_pie, use_container_width=True)
        
        # Danceability 直方圖
        if chart_dance := plot_histogram(df, 'danceability', '舞蹈指數 (Danceability)'):
            st.altair_chart(chart_dance, use_container_width=True)
        
    # --- 5b. 單曲分析頁面 ---
    else:
        # 使用 .loc 提高效率和避免警告
        # 確保 display_name 存在才進行過濾
        if 'display_name' in df.columns:
            song_data_rows = df.loc[df['display_name'] == selected_song]
            if not song_data_rows.empty:
                song_data = song_data_rows.iloc[0]
            else:
                st.error(f"錯誤：無法在數據中找到歌曲 '{selected_song}'。")
                return # 停止執行
        else:
             st.error("錯誤：數據中缺少 'display_name' 欄位。")
             return # 停止執行
        
        st.title(f"🎵 {song_data.get('track_name', 'N/A')}") # 使用 .get() 更安全
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
            # 使用本地變數 ai_available
            if ai_available and pd.notna(song_data.get('ai_theme')) and song_data.get('ai_theme') not in ['SKIPPED', 'ERROR']:
                st.info(f"**AI 主題 (Theme):**\n{song_data.get('ai_theme', 'N/A')}") 
                st.warning(f"**AI 情緒類別 (Category):** {song_data.get('ai_sentiment_category', 'N/A')}\n"
                           f"*(原始情緒: {song_data.get('ai_sentiment', 'N/A')})*")
                st.markdown("**AI 綜合筆記 (Notes):**")
                st.write(song_data.get('ai_notes', 'N/A'))
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
                'ai_sentiment_category' 
            ]
            
            # 動態排除所有臨時欄位
            temp_cols = [col for col in song_data.index if col.endswith('_bin') or col == 'key_name' or col == 'key_scale_combined']
            manual_cols.extend(temp_cols)
            
            other_fields = song_data.drop(labels=list(set(manual_cols)), errors='ignore') 
            other_fields_with_data = other_fields.dropna()
            
            if not other_fields_with_data.empty:
                st.dataframe(other_fields_with_data, use_container_width=True)
            else:
                st.info("此歌曲沒有其他可用的 (Tonal/AcousticBrainz) 資料。")

# --- 6. 執行 Main ---
if __name__ == "__main__":
    # 不再依賴 session_state 進行初始化，main() 函數內部會處理
    main()


