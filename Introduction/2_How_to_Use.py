import streamlit as st

st.title("How to Use")

st.subheader("Installation (from GitHub)")

st.markdown(
    """
    1. 進入終端，將位置切換至本專案資料夾。
    """
) 

st.code(
    """
    cd <目標資料夾的絕對位置>
    """,
    language = "python"
)

st.markdown(
    """
    2. 終端輸入下列程式碼以配置專案環境 (環境名稱為 house-price-prediction)。
    """
) 

st.code(
    """
    conda env create -f environment.yml
    """,
    language = "python"
)

st.markdown(
    """
    3. 終端輸入下列程式碼以啟動專案環境 (環境名稱為 house-price-prediction)。
    """
) 

st.code(
    """
    conda activate house-price-prediction
    """,
    language = "python"
)

st.markdown(
    """
    4. 終端輸入下列程式碼以啟動專案。
    """
) 

st.code(
    """
    streamlit run Home.py
    """,
    language = "python"
)

st.subheader("Get Started")

st.markdown(
    """
    1. 由側邊欄點擊進入 House Query 頁面，在「房屋檢索」頁籤中勾選「開啟屬性篩選」並依序輸入您的房屋偏好。
    2. 篩選完成後可能出現多個房屋物件，請選擇其中有興趣之物件，記下物件最左側之房屋編號 (無欄位名稱)，並將該編號輸入至側邊欄中「欲分析之房屋編號」欄位。
    3. 在側邊欄中依自身條件填入剩餘欄位，填寫完畢即可點擊「更新儀表板」送出表單並執行分析。
    4. 等待畫面中出現「分析完成」訊息後，請點擊進入「房屋分析儀表板」頁籤並查看分析結果 (收起側邊欄可獲得最佳瀏覽儀表板體驗)。
    5. 點擊周邊設施下拉選項即可查看周邊設施名稱，開啟「地圖顯示」則能在畫面中央的地圖上查看設施分布情形。
    """
)
