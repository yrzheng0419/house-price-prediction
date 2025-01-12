import streamlit as st
import pandas as pd
import h3
import joblib
import pydeck as pdk
import geopandas as gpd
import shapely.geometry
from datetime import datetime

import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("House Query")

#-------------cache space start---------------#

# Set cache to those params which doesn't interact with users.

# set url
DATA_URL = '.\data\data.csv'
DATA_CHINESE_URL = '.\data\data_chinese.csv'

DATA_NIMBY_URL = '.\data\\nimby_st.csv'
DATA_GREEN_URL = '.\data\green_st.csv'
DATA_SCHOOL_URL = '.\data\school_st.csv'
DATA_MEDICAL_URL = '.\data\medical_st.csv'
DATA_BUS_URL = '.\data\\bus_st.csv'
DATA_MRT_URL = '.\data\mrt_st.csv'
DATA_TRAIN_URL = '.\data\\train_st.csv'

MODEL_BUY_URL = '.\model\\rf_buy_bst.joblib'
MODEL_RENT_URL = '.\model\\rf_rent_bst.joblib'

# def get_data func
@st.cache_data
def get_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

# def get_model func
@st.cache_resource
def get_model(url: str) -> pd.DataFrame:
    return joblib.load(url)

# get data & model
dat = get_data(DATA_URL)
dat_chinese = get_data(DATA_CHINESE_URL)

dat_nimby = get_data(DATA_NIMBY_URL)
dat_green = get_data(DATA_GREEN_URL)
dat_school = get_data(DATA_SCHOOL_URL)
dat_medical = get_data(DATA_MEDICAL_URL)
dat_bus = get_data(DATA_BUS_URL)
dat_mrt = get_data(DATA_MRT_URL)
dat_train = get_data(DATA_TRAIN_URL)

model_buy = get_model(MODEL_BUY_URL)
model_rent = get_model(MODEL_RENT_URL)

#--------------cash space end--------------#

#--------------session_state & sidebar start---------------#

# All params to initialize the dashboard (params interacting with users).

if "index_on" not in st.session_state:
    st.session_state.index_on = 0
if "expect_on" not in st.session_state:
    st.session_state.expect_on = 15000000
if "income_on" not in st.session_state:
    st.session_state.income_on = 2000000
if "debt_on" not in st.session_state:
    st.session_state.debt_on = 10000000
if "rate_on" not in st.session_state:
    st.session_state.rate_on = 1.775
if "year_on" not in st.session_state:
    st.session_state.year_on = 40
if "buffer_on" not in st.session_state:
    st.session_state.buffer_on = 5
if "medical_arc" not in st.session_state:
    st.session_state.medical_arc = False
if "school_arc" not in st.session_state:
    st.session_state.school_arc = False
if "green_arc" not in st.session_state:
    st.session_state.green_arc = False

st.sidebar.subheader("請填入以下資訊執行分析")

def on_button_click():
    with st.spinner("更新中..."):
        st.session_state.index_on = index
        st.session_state.expect_on = expect
        st.session_state.income_on = income
        st.session_state.debt_on = debt
        st.session_state.rate_on = rate
        st.session_state.year_on = year
        st.session_state.buffer_on = buffer
    st.success("分析完成！請至房屋分析儀表板查看結果")

with st.sidebar:
    index = st.number_input("欲分析之房屋編號", 0)
    expect = st.number_input("預期房屋價格", 0, 9999999999, 15000000)
    income = st.number_input("年收入(預設全年總薪資中位數)", 0, 9999999999, 2000000)
    debt = st.number_input("貸款本金", 0, 9999999999, 10000000)
    rate = st.number_input("預估每月單一利率 (%)", 0.0, 100.0, 1.775, format="%.3f")
    year = st.number_input("還款期數", 0, 100, 40)
    buffer = st.slider("生活圈範圍 (半徑87m/單位)", 0, 10, 5)
    
    st.button("更新儀表板", on_click = on_button_click)

#--------------session_state & sidebar end---------------#

#--------------query func start--------------#

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("開啟屬性篩選 (點擊欄位名稱可依該欄位數值排序)")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"範圍 - {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"範圍 - {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"範圍 - {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    st.success(f"共搜尋到 {df.shape[0]} 筆符合條件的房屋物件，篩選完成後請將資料框最左側之房屋編號填入側邊欄。")
    return df

#--------------query func end--------------#

#---------------rent correlation start---------------#

# Evaluate the variation trend of house price in the future with the VIP of the RF model.

rent_model_imp = {
    "district": 0.411731,
    "rich_poor_gap": 0.102663,
    "age": 0.049007,
    "a65_a0a14_rat": 0.047942,
    "income_median": 0.043363,
    "medical": 0.023114,
    "urban_index": 0.015536,
    "crime": 0.007869
}

dat_groupby_rent = (
    dat[['district', 'rich_poor_gap', 'age', 'a65_a0a14_rat', 'income_median', 'medical', 'urban_index', 'crime']]
    .groupby('district', as_index=False)
    .sum()
)

def rent_corr_formula(dat_avg: pd.DataFrame, dict_imp: dict, sample: pd.DataFrame):
    sum = 0
    sample = sample.reset_index(drop = True)
    district = sample.loc[0, 'district'].item()
    district_avg = dat_avg.loc[lambda x: x['district'] == district].reset_index(drop = True)

    # rich_poor_gap O
    if sample.loc[0, 'rich_poor_gap'].item() / district_avg.loc[0, 'rich_poor_gap'].item() > 1:
        sum += abs(float(sample.loc[0, 'rich_poor_gap'].item() / district_avg.loc[0, 'rich_poor_gap'].item()) - 1) * dict_imp.get('rich_poor_gap')
    else:
        sum += abs(float(sample.loc[0, 'rich_poor_gap'].item() / district_avg.loc[0, 'rich_poor_gap'].item()) - 1) * (-1) * dict_imp.get('rich_poor_gap')

    # age
    if sample.loc[0, 'age'].item() / district_avg.loc[0, 'age'].item() > 1:
        sum += abs(float(sample.loc[0, 'age'].item() / district_avg.loc[0, 'age'].item()) - 1) * dict_imp.get('age')
    else:
        sum += abs(float(sample.loc[0, 'age'].item() / district_avg.loc[0, 'age'].item()) - 1) * (-1) * dict_imp.get('age')

    # a65_a0a14_rat
    if sample.loc[0, 'a65_a0a14_rat'].item() / district_avg.loc[0, 'a65_a0a14_rat'].item() > 1:
        sum += abs(float(sample.loc[0, 'a65_a0a14_rat'].item() / district_avg.loc[0, 'a65_a0a14_rat'].item()) - 1) * dict_imp.get('a65_a0a14_rat')
    else:
        sum += abs(float(sample.loc[0, 'a65_a0a14_rat'].item() / district_avg.loc[0, 'a65_a0a14_rat'].item()) - 1) * (-1) * dict_imp.get('a65_a0a14_rat')

    # income_medium
    if sample.loc[0, 'income_median'].item() / district_avg.loc[0, 'income_median'].item() > 1:
        sum += abs(float(sample.loc[0, 'income_median'].item() / district_avg.loc[0, 'income_median'].item()) - 1) * dict_imp.get('income_median')
    else:
        sum += abs(float(sample.loc[0, 'income_median'].item() / district_avg.loc[0, 'income_median'].item()) - 1) * (-1) * dict_imp.get('income_median')

    # medical
    if sample.loc[0, 'medical'].item() / district_avg.loc[0, 'medical'].item() > 1:
        sum += abs(float(sample.loc[0, 'medical'].item() / district_avg.loc[0, 'medical'].item()) - 1) * dict_imp.get('medical')
    else:
        sum += abs(float(sample.loc[0, 'medical'].item() / district_avg.loc[0, 'medical'].item()) - 1) * (-1) * dict_imp.get('medical')

    # urban_index
    if sample.loc[0, 'urban_index'].item() / district_avg.loc[0, 'urban_index'].item() > 1:
        sum += abs(float(sample.loc[0, 'urban_index'].item() / district_avg.loc[0, 'urban_index'].item()) - 1) * dict_imp.get('urban_index')
    else:
        sum += abs(float(sample.loc[0, 'urban_index'].item() / district_avg.loc[0, 'urban_index'].item()) - 1) * (-1) * dict_imp.get('urban_index')

    # crime
    if sample.loc[0, 'crime'].item() / district_avg.loc[0, 'crime'].item() > 1:
        sum += abs(float(sample.loc[0, 'crime'].item() / district_avg.loc[0, 'crime'].item()) - 1) * dict_imp.get('crime')
    else:
        sum += abs(float(sample.loc[0, 'crime'].item() / district_avg.loc[0, 'crime'].item()) - 1) * (-1) * dict_imp.get('crime')

    return sum

#---------------rent correlation end---------------#

#---------------dashboard params start-----------------#

index_on = st.session_state.index_on

CENTER = {
    "lat": dat.iloc[index_on, 2],
    "lng": dat.iloc[index_on, 1]
}
CENTER_HEX = dat.loc[index_on, 'hex_id']
LIFE_CIRCLE = h3.grid_disk(CENTER_HEX, st.session_state.buffer_on)

# get sample by dataframe
sample_buy = dat.drop(['hex_id', 'neighbor_avg_ping', 'ping'], axis = 1).loc[[index_on]]
sample_rent = dat.drop(['hex_id', 'mean_rent', 'neighbor_avg_ping', 'ping'], axis = 1).loc[[index_on]]

# raw buy price by float
buy_price = model_buy.predict(sample_buy)
buy_price = round(float(buy_price.item() * sample_buy.loc[index_on, 'area'].item()) * 10000)

# raw rent price by float
rent_price = model_rent.predict(sample_rent)
rent_price = round(float(rent_price.item() * sample_rent.loc[index_on, 'area'].item()))

# calculate rent corr
rent_corr = round(rent_corr_formula(dat_groupby_rent, rent_model_imp, sample_rent), 2)

# data for filter
sample_buy_chinese = dat_chinese.loc[[index_on]]

# data for dashboard
hex_id = dat.loc[index_on, 'hex_id']

green_include = dat_green.iloc[dat_green.isin(LIFE_CIRCLE)['hex_id'].tolist()]
school_include = dat_school.iloc[dat_school.isin(LIFE_CIRCLE)['hex_id'].tolist()]
medical_include = dat_medical.iloc[dat_medical.isin(LIFE_CIRCLE)['hex_id'].tolist()]

nimby_include = dat_nimby.iloc[dat_nimby.isin(LIFE_CIRCLE)['hex_id'].tolist()].shape[0]
bus_include = dat_bus.iloc[dat_bus.isin(LIFE_CIRCLE)['hex_id'].tolist()].shape[0]
mrt_include = dat_mrt.iloc[dat_mrt.isin(LIFE_CIRCLE)['hex_id'].tolist()].shape[0]
train_include = dat_train.iloc[dat_train.isin(LIFE_CIRCLE)['hex_id'].tolist()].shape[0]

#---------------dashboard params end-----------------#

#---------------map start------------------#

view_state = pdk.ViewState(
    latitude = CENTER['lat'],
    longitude = CENTER['lng'],
    zoom=14,
    min_zoom = 12,
    max_zoom = 24,
    pitch=45,
)

# get life cycle geopandas
def life_cycle_to_gpd(hexagons: list) -> gpd.GeoDataFrame:
    hexagon_geometries = [
        shapely.geometry.Polygon(h3.cell_to_boundary(hexagon))
        for hexagon in hexagons
    ]

    # Create a GeoDataFrame to store hexagon data
    hexagon_df = gpd.GeoDataFrame({'hex_id': hexagons, 'geometry': hexagon_geometries})

    return hexagon_df

h3_grid = life_cycle_to_gpd(LIFE_CIRCLE)

life_cycle_layer = pdk.Layer(
    "H3HexagonLayer",
    h3_grid,
    pickable = False,
    stroked = True,
    filled = False,
    extruded = False,
    get_hexagon = "hex_id",
    get_line_color = [255, 255, 255],
    line_width_min_pixels = 1,
)

# get facility around
WHITE_RGB = [255, 255, 255]
RED_RGB = [240, 100, 0]
GREEN_RGB = [0, 255, 0]
BLUE_RGB = [0, 0, 255]

medical_arc_dat = life_cycle_to_gpd(medical_include.loc[:, 'hex_id'].tolist())
medical_arc_layer = pdk.Layer(
    "H3HexagonLayer",
    medical_arc_dat,
    pickable=False,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="hex_id",
    get_fill_color=RED_RGB,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

school_arc_dat = life_cycle_to_gpd(school_include.loc[:, 'hex_id'].tolist())
school_arc_layer = pdk.Layer(
    "H3HexagonLayer",
    school_arc_dat,
    pickable=False,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="hex_id",
    get_fill_color=BLUE_RGB,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

green_arc_dat = life_cycle_to_gpd(green_include.loc[:, 'hex_id'].tolist())
green_arc_layer = pdk.Layer(
    "H3HexagonLayer",
    green_arc_dat,
    pickable=False,
    stroked=True,
    filled=True,
    extruded=False,
    get_hexagon="hex_id",
    get_fill_color=GREEN_RGB,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

def get_arcmap() -> pdk.Deck:
    arc_list = [life_cycle_layer]
    if st.session_state.medical_arc:
        arc_list.insert(0, medical_arc_layer)
    if st.session_state.school_arc:
        arc_list.insert(0, school_arc_layer)
    if st.session_state.green_arc:
        arc_list.insert(0, green_arc_layer)

    map = pdk.Deck(
        layers = arc_list,
        initial_view_state = view_state,
        tooltip={
            "html": """
                    <b>Name:</b> {name}<br>
                    <b>Location:</b> ({lat}, {lng})<br>
            """,
            "style": {"color": "white"},
        }
    )
    return map

if "arc_map" not in st.session_state:
    st.session_state.arc_map = get_arcmap()

#------------------map end----------------------#

#------------------dashboard start---------------------#

tab1, tab2 = st.tabs(["房屋檢索", "房屋分析儀表板 (隱藏側邊欄以獲得最佳體驗)"])

with tab1:
    st.dataframe(filter_dataframe(dat_chinese))

with tab2:
    dashboard = st.container()

    with dashboard:
        dash1, dash2, dash3 = st.columns([0.25, 0.45, 0.3], border=False)

        with dash1:
            # 分析時間
            current_time = datetime.now().strftime("%Y-%m-%d")

            st.metric(label = '⏱️ 分析日期', value = current_time)

            # 預測房價
            hou_price = buy_price * 10000
            st.metric(label = '預測房價 (NT) (測試版)', value = f'{round(hou_price)} NT', border = False, help = "Model v1.0.0")

            # 預測每月房租
            st.metric(label = '預測每月房租約 (NT)', value = f'{round(rent_price)} NT', border = False, help = "Model v1.0.0")

            # 房屋基本資訊 (text_table)
            sample_info = pd.DataFrame(
                {
                    "基本屬性": ['房屋編號', '建物型態', '交易類型', '交易物件', '屋齡', '移轉/總樓層數', '房屋格局', "周邊交通點位數量"],
                    "屬性資訊": [f'#{index_on}',
                                f'{sample_buy.loc[index_on, 'type']}',
                                f'{sample_buy.loc[index_on, 'class']}',
                                f'{sample_buy.loc[index_on, 'land']} 土地 {sample_buy.loc[index_on, 'building']} 建物 {sample_buy.loc[index_on, 'parking']} 車位',
                                f'{sample_buy.loc[index_on, 'age']} 年',
                                f'{round(float(sample_buy.loc[index_on, 'floor_ratio'].item() * sample_buy.loc[index_on, 't_floor'].item()))} 樓/{sample_buy.loc[index_on, 't_floor']} 樓',
                                f'{sample_buy.loc[index_on, 'room']} 房 {sample_buy.loc[index_on, 'living_room']} 廳 {sample_buy.loc[index_on, 'bathroom']} 衛 {sample_buy.loc[index_on, 'partition']} 隔間',
                                f"{bus_include + mrt_include + train_include} 個"]
                }
            )
            st.data_editor(sample_info, hide_index = True, use_container_width = True)

            nimby_col, crime_col = st.columns([0.4, 0.6])

            with nimby_col:
                st.metric(label = "嫌惡設施數量", value = f'{nimby_include}個')
            with crime_col:
                crime_signal = "+" if sample_buy.loc[index_on, 'crime'].item() < 7 else "-"
                crime_message = "治安相較穩定" if sample_buy.loc[index_on, 'crime'].item() < 7 else "治安相對較差"
                st.metric(label = "累積犯罪次數", value = f"{sample_buy.loc[index_on, 'crime'].item()} 次", delta = f"{crime_signal} {crime_message}",
                            help = "1101-1143 季度之台北市累積犯罪次數中位數為 7/hexagon")

        with dash2:
            dash21, dash22, dash23 = st.columns(3)

            # 房價所得比
            with dash21:
                hou_income_ratio = round(st.session_state.expect_on/st.session_state.income_on, 2)
                hi_signal = "-" if hou_income_ratio > 10 else "+"
                hi_message = "購屋負擔較重" if hou_income_ratio > 10 else "購屋負擔較輕"
                st.metric(label = "房價所得比", value = hou_income_ratio, border = True,
                            delta = f"{hi_signal}{hi_message}")

            # 每月繳款
            with dash22:
                n = (st.session_state.year_on * 12)
                rate_month = (st.session_state.rate_on / 12) /100
                repay = round(st.session_state.debt_on * ((rate_month * ((1 + rate_month)**n)) / (((1 + rate_month)**n) - 1)))
                repay_signal = "+" if (repay/(st.session_state.income_on/12)) <0.33 else "-"
                st.metric(label = "每月繳款 (NT)", value = f'{repay}', delta = f"{repay_signal}佔所得{round((repay/(st.session_state.income_on/12))*100, 1)}%",
                            help = "理想情況為每月房貸本息攤還的金額不要超過家庭收入的三分之一。", border = True)

            # 房價租金比
            with dash23:
                br_dict = {
                    "中正區": 55.5,
                    "南港區": 50.3,
                    "文山區": 49.4,
                    "大安區": 49.0,
                    "松山區": 44.6,
                    "士林區": 42.7,
                    "北投區": 41.2,
                    "中山區": 38.9,
                    "信義區": 38.6,
                    "內湖區": 38.0,
                    "萬華區": 34.3,
                    "大同區": 34.0
                }

                buy_rent_ratio = round((buy_price*10000)/(rent_price*12))
                brr_signal = "+" if (buy_rent_ratio - br_dict.get(sample_buy_chinese.loc[index_on, '行政區'])) < 0 else "-"
                brr_message = "推薦購屋" if (buy_rent_ratio - br_dict.get(sample_buy_chinese.loc[index_on, '行政區'])) < 0 else "推薦租屋"
                
                st.metric(label="房價租金比", value = buy_rent_ratio, delta = f"{brr_signal}{brr_message}",
                            help = f"判斷條件參考內政部不動產交易時價查詢網之 2024 年 1-9 月台北市房價租金比，其中{sample_buy_chinese.loc[index_on, '行政區']}之房價租金比為 {br_dict.get(sample_buy_chinese.loc[index_on, '行政區'])}。", border = True)

            st.pydeck_chart(get_arcmap())

            st.caption("※ 白色六邊形為房屋所在地，空心六邊形為生活圈覆蓋範圍。")

        with dash3:
            # 房價所得比 & 每月需繳貸款金額
            rp_col, rc_col = st.columns(2)

            # 年投資報酬率
            with rp_col:
                rp_signal = "-" if rent_corr < 4 else "+"
                rp_message = "不適合投資" if rent_corr < 4 else "適合投資"
                st.metric(label = "年投資報酬率", value = f"{round((rent_price*12*100)/(buy_price*10000), 2)} %", delta = f"{rp_signal} {rp_message}", border = True, help = "參考大部分房屋網站建議為 4%")

            # 漲跌空間
            with rc_col:
                rc_signal = "-" if rent_corr < 0 else "+"
                rc_message = "未來可能下跌" if rent_corr < 0 else "未來可能上漲"
                st.metric(label="漲跌空間", value = f"{rent_corr}%", delta = f"{rc_signal} {rc_message}", border = True,
                            help = "當周圍物件")

            st.markdown("**以下四項指標用於多物件橫向比較：**")
            
            # 貧富差距指數 & 鬧區指數
            rpg_col, bustle_col = st.columns(2)

            with rpg_col:
                rpg_value = round(float(sample_buy.loc[index_on, 'rich_poor_gap'].item()), 2)
                st.metric(label = "貧富差距指數", value = rpg_value,
                            help = "貧富差距越大可能代表當地市容較混亂，計算方式為每一個 h3 網格之年收入 IQR，其服從標準常態分布。")
            with bustle_col:
                st.metric(label = "鬧區指數", value = round(float(sample_buy.loc[index_on, 'urban_index'].item()), 2), help = "鬧區指數越大可能代表假日周邊較吵雜。")

            # 老化指數 & 收入中位數
            ai_col, im_col = st.columns(2)

            with ai_col:
                st.metric(label = "老化指數", value = round(float(sample_buy.loc[index_on, 'age'].item()), 2), help = "老化指數較高可能代表當地的扶老政策較完善(或資源較集中)。")
            with im_col:
                st.metric(label = "年收入中位數(萬)", value = round(float(sample_buy.loc[index_on, 'income_median'].item()), 2), help = "年收入中位數可以用於預期鄰里的生活水準。")

            st.divider()

            # ---------------------------------------周邊設施
            st.markdown("**周邊設施**")

            def update_arc_state():
                st.session_state.medical_arc = False if medical_arc else True
                st.session_state.school_arc = False if medical_arc else True
                st.session_state.green_arc = False if medical_arc else True
                st.session_state.arc_map = get_arcmap()

            with st.expander(f"💊 點擊查看周邊 **{medical_include.shape[0]}** 家醫療設施"):
                medical_arc = st.toggle("地圖顯示", key = "medical_arc", on_change = update_arc_state)
                for i in medical_include.to_dict(orient = "records"):
                    st.text(f"• {i['name']}")
            with st.expander(f"🏫 點擊查看周邊 **{school_include.shape[0]}** 個教育設施"):
                school_arc = st.toggle("地圖顯示", key = "school_arc", on_change = update_arc_state)
                for i in school_include.to_dict(orient = "records"):
                    st.text(f"• {i['name']}")
            with st.expander(f"🌳 點擊查看周邊 **{green_include.shape[0]}** 片綠地"):
                green_arc = st.toggle("地圖顯示", key = "green_arc", on_change = update_arc_state)
                for i in green_include.to_dict(orient = "records"):
                    st.text(f"• {i['name']}")


#------------------dashboard end---------------------#

#==========================reference=======================#

# https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/#code-section-3-writing-conditionals-for-different-column-types
# https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/
# https://h3-snow.streamlit.app/
# https://www.atyun.com/57621.html