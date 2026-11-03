from __future__ import annotations

import io
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


st.set_page_config(page_title="따릉이 시간대·계절 예측", page_icon="🚲", layout="wide")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family:'DM Sans','Noto Sans KR',sans-serif; }
      .stApp { background:#f4f6f9; color:#1c2738; }
      [data-testid="stSidebar"] { background:#fff; border-right:1px solid #e3e9f0; }
      .hero { display:flex;align-items:center;gap:12px;margin:0 0 2px; }
      .hero h1 { font-size:1.7rem;letter-spacing:-.04em;margin:0;color:#172033; }
      .hero-icon { width:42px;height:42px;border-radius:14px;background:#e8f5ee;display:flex;align-items:center;justify-content:center;font-size:23px; }
      .subtle { color:#718096;font-size:.9rem;margin:4px 0 15px; }
      .pill { display:inline-block;border-radius:20px;padding:5px 10px;background:#e8f5ee;color:#21744c;font-size:.75rem;font-weight:700; }
      .pill.demo { background:#fff2d9;color:#956315; }
      .note { border:1px solid #f2d491;background:#fff9eb;border-radius:12px;padding:10px 14px;color:#765819;font-size:.83rem;margin:10px 0 15px; }
      .section-title { font-size:1.08rem;font-weight:750;margin:18px 0 9px;color:#1c2738; }
      .footnote { color:#738095;font-size:.77rem;margin-top:12px; }
      div[data-testid="stMetric"] { background:#fff;border:1px solid #e1e7ef;border-radius:13px;padding:15px 17px;box-shadow:0 2px 8px rgba(24,34,48,.025); }
      div[data-testid="stMetricLabel"] { color:#708096; }
      div[data-testid="stMetricValue"] { color:#1d2b40;font-size:1.48rem; }
      div[data-testid="stDataFrame"] { border:1px solid #e1e7ef;border-radius:12px;overflow:hidden;background:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

HISTORY_URL = "https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do"
HOURLY_URL = "https://data.seoul.go.kr/dataList/OA-15245/F/1/datasetView.do"
REALTIME_URL = "https://data.seoul.go.kr/dataList/OA-15493/A/1/datasetView.do"
MAX_MODEL_STATIONS = 24
SEASON_NAMES = {1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을", 12: "겨울"}
SEASON_CODE = {"봄": 0, "여름": 1, "가을": 2, "겨울": 3}


def key_text(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^\s*\d+\s*[.)_-]?\s*", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def find_column(columns, aliases: list[str]) -> str | None:
    normalized = {key_text(col): col for col in columns}
    for alias in aliases:
        if key_text(alias) in normalized:
            return normalized[key_text(alias)]
    for key, original in normalized.items():
        if any(key_text(alias) in key for alias in aliases if len(key_text(alias)) >= 3):
            return original
    return None


def read_csv(uploaded) -> pd.DataFrame:
    content = uploaded.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            frame = pd.read_csv(io.BytesIO(content), encoding=encoding)
            if len(frame.columns) == 1 and b";" in content[:4096]:
                frame = pd.read_csv(io.BytesIO(content), encoding=encoding, sep=";")
            return frame
        except Exception as exc:
            last_error = exc
    raise ValueError(f"CSV를 읽지 못했습니다: {last_error}")


def make_demo_hourly() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(48)
    stations = [
        ("101", "광화문역 7번출구", 24, 8, 18),
        ("102", "시청역 1번출구", 22, 15, 12),
        ("103", "서울역 2번출구", 30, 24, 23),
        ("104", "종각역 5번출구", 20, 5, 15),
        ("105", "을지로입구역 3번출구", 26, 19, 14),
        ("106", "경복궁역 4번출구", 18, 11, 9),
    ]
    end = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=1)
    hours = pd.date_range(end - pd.Timedelta(days=419), end, freq="h")
    records = []
    base_profile = [0.12, 0.08, 0.05, 0.04, 0.03, 0.05, 0.18, 0.38, 0.52, 0.32, 0.24, 0.25,
                    0.31, 0.24, 0.22, 0.28, 0.46, 0.65, 0.72, 0.51, 0.34, 0.25, 0.18, 0.13]
    season_factor = {"봄": 1.0, "여름": 1.16, "가을": 1.05, "겨울": 0.68}
    for station_id, name, capacity, current, base in stations:
        for ts in hours:
            season = SEASON_NAMES[ts.month]
            commute = 1.0 if ts.weekday() >= 5 else 1.0
            if ts.weekday() < 5 and ts.hour in (7, 8, 17, 18, 19):
                commute = 1.65
            rent_mean = base * base_profile[ts.hour] * commute * season_factor[season]
            return_mean = base * base_profile[(ts.hour - 2) % 24] * commute * season_factor[season] * (1.02 if ts.weekday() < 5 else 0.96)
            rentals = int(rng.poisson(max(0.025, rent_mean)))
            returns = int(rng.poisson(max(0.025, return_mean)))
            records.append((ts, station_id, name, rentals, returns))
    history = pd.DataFrame(records, columns=["timestamp", "station_id", "station_name", "rentals", "returns"])
    stock = pd.DataFrame(
        {
            "station_id": [row[0] for row in stations], "station_name": [row[1] for row in stations],
            "capacity": [row[2] for row in stations], "current_bikes": [row[3] for row in stations],
            "latitude": [37.5710, 37.5657, 37.5569, 37.5700, 37.5660, 37.5760],
            "longitude": [126.9769, 126.9771, 126.9720, 126.9850, 126.9820, 126.9730],
        }
    )
    return history, stock


def standardize_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    ts_col = find_column(raw.columns, ["timestamp", "대여일시", "기준시간", "date_hour", "datetime"])
    date_col = find_column(raw.columns, ["대여일자", "date"])
    hour_col = find_column(raw.columns, ["대여시간", "시간", "hour"])
    id_col = find_column(raw.columns, ["station_id", "대여소번호", "대여대여소번호", "대여대여소ID"])
    name_col = find_column(raw.columns, ["station_name", "대여소명", "대여대여소명"])
    rent_col = find_column(raw.columns, ["rentals", "대여건수", "대여수", "대여량"])
    return_col = find_column(raw.columns, ["returns", "반납건수", "반납수", "반납량"])
    if ts_col:
        timestamps = pd.to_datetime(raw[ts_col], errors="coerce")
    elif date_col and hour_col:
        date_values = raw[date_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        compact = date_values.str.replace(r"[^0-9]", "", regex=True)
        parsed = pd.to_datetime(date_values, errors="coerce")
        eight = compact.str.len().eq(8)
        parsed.loc[eight] = pd.to_datetime(compact.loc[eight], format="%Y%m%d", errors="coerce")
        hour_values = pd.to_numeric(raw[hour_col].astype("string").str.extract(r"(\d{1,2})")[0], errors="coerce")
        timestamps = parsed + pd.to_timedelta(hour_values, unit="h")
    else:
        timestamps = None
    if timestamps is None or not (id_col or name_col) or not rent_col or not return_col:
        raise ValueError(
            "시간대 집계 열을 찾지 못했습니다. timestamp, station_id, station_name, rentals, returns 형식의 CSV를 올려주세요. "
            "원본 대여이력은 prepare_hourly_data.py로 먼저 변환할 수 있습니다."
        )
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(timestamps, errors="coerce").dt.floor("h")
    out["station_id"] = raw[id_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True) if id_col else raw[name_col].astype("string").str.strip()
    out["station_name"] = raw[name_col].astype("string").str.strip().fillna(out["station_id"]) if name_col else out["station_id"]
    out["rentals"] = pd.to_numeric(raw[rent_col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    out["returns"] = pd.to_numeric(raw[return_col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    out = out.dropna(subset=["timestamp", "station_id", "rentals", "returns"])
    out = out[(out["rentals"] >= 0) & (out["returns"] >= 0)]
    if out.empty:
        raise ValueError("유효한 시간·대여소·대여·반납 행이 없습니다.")
    return out.groupby(["timestamp", "station_id", "station_name"], as_index=False)[["rentals", "returns"]].sum()


def standardize_stock(raw: pd.DataFrame) -> pd.DataFrame:
    name_col = find_column(raw.columns, ["station_name", "대여소명", "대여소", "stationName"])
    id_col = find_column(raw.columns, ["station_id", "대여소ID", "대여소번호", "stationId"])
    bikes_col = find_column(raw.columns, ["current_bikes", "자전거주차총건수", "현재자전거수", "대여가능수", "parkingBikeTotCnt"])
    capacity_col = find_column(raw.columns, ["capacity", "거치대개수", "거치대수", "rackTotCnt"])
    lat_col = find_column(raw.columns, ["latitude", "위도", "stationLatitude"])
    lon_col = find_column(raw.columns, ["longitude", "경도", "stationLongitude"])
    if not (name_col or id_col) or not bikes_col:
        raise ValueError("재고 CSV에는 station_name(또는 station_id)과 current_bikes 열이 필요합니다.")
    result = pd.DataFrame()
    result["station_id"] = raw[id_col].astype("string").str.strip() if id_col else ""
    result["station_name"] = raw[name_col].astype("string").str.strip() if name_col else result["station_id"]
    result["station_name"] = result["station_name"].fillna(result["station_id"])
    result["current_bikes"] = pd.to_numeric(raw[bikes_col], errors="coerce")
    result["capacity"] = pd.to_numeric(raw[capacity_col], errors="coerce") if capacity_col else np.nan
    result["latitude"] = pd.to_numeric(raw[lat_col], errors="coerce") if lat_col else np.nan
    result["longitude"] = pd.to_numeric(raw[lon_col], errors="coerce") if lon_col else np.nan
    return result.dropna(subset=["current_bikes"]).drop_duplicates("station_name", keep="last")


def fetch_realtime_stock(api_key: str) -> pd.DataFrame:
    rows, start = [], 1
    while start <= 10000:
        end = start + 999
        url = f"https://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start}/{end}"
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        payload = response.json()
        service = payload.get("rentBikeStatus")
        if service is None:
            raise ValueError(payload.get("RESULT", {}).get("MESSAGE", "서울시 API 응답 형식을 확인하지 못했습니다."))
        rows.extend(service.get("row", []))
        if end >= int(service.get("list_total_count", len(rows))) or not service.get("row"):
            break
        start = end + 1
    if not rows:
        raise ValueError("실시간 대여소 응답이 없습니다.")
    return standardize_stock(pd.DataFrame(rows))


def season_of(month: int) -> str:
    return SEASON_NAMES[int(month)]


def feature_table(frame: pd.DataFrame, one_hot_columns=None):
    features = pd.DataFrame(index=frame.index)
    features["station_id"] = frame["station_id"].astype(str)
    features["season"] = frame["season"].astype(str)
    features["hour"] = frame["timestamp"].dt.hour.astype(int)
    features["weekday"] = frame["timestamp"].dt.weekday.astype(int)
    features["is_weekend"] = (features["weekday"] >= 5).astype(int)
    features["is_commute"] = features["hour"].isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    features["is_night"] = features["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    month = frame["timestamp"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features = pd.get_dummies(features, columns=["station_id", "season"], dtype=float)
    if one_hot_columns is not None:
        features = features.reindex(columns=one_hot_columns, fill_value=0.0)
    return features


def make_panel(history: pd.DataFrame, focus_id: str):
    end = pd.Timestamp(history["timestamp"].max()).floor("h")
    start = max(pd.Timestamp(history["timestamp"].min()).floor("h"), end - pd.Timedelta(days=729))
    totals = history.groupby(["station_id", "station_name"])["rentals"].sum().sort_values(ascending=False)
    ranked = [str(index[0]) for index in totals.index[:MAX_MODEL_STATIONS]]
    if focus_id not in ranked:
        ranked = [focus_id] + ranked[: MAX_MODEL_STATIONS - 1]
    name_map = history.sort_values("timestamp").groupby("station_id")["station_name"].last().to_dict()
    grid = pd.date_range(start, end, freq="h")
    parts = []
    source = history.copy()
    source["station_id"] = source["station_id"].astype(str)
    for station_id in ranked:
        station = source[source["station_id"] == station_id].groupby("timestamp")[["rentals", "returns"]].sum()
        if station.empty:
            continue
        station = station.reindex(grid, fill_value=0).reset_index(names="timestamp")
        station["station_id"] = station_id
        station["station_name"] = name_map.get(station_id, station_id)
        station["season"] = station["timestamp"].dt.month.map(SEASON_NAMES)
        parts.append(station)
    if not parts:
        raise ValueError("선택 대여소에 집계할 시간별 이력이 없습니다.")
    return pd.concat(parts, ignore_index=True), end


@st.cache_data(show_spinner=False, ttl=900, max_entries=3)
def train_and_forecast(history: pd.DataFrame, focus_id: str, horizon: int):
    panel, cutoff = make_panel(history, focus_id)
    x = feature_table(panel)
    x_columns = list(x.columns)
    y = panel[["rentals", "returns"]].to_numpy(dtype=float)
    unique_hours = np.sort(panel["timestamp"].unique())
    if len(unique_hours) < 24 * 30:
        raise ValueError("최소 30일 이상 시간별 자료가 필요합니다. 계절 비교에는 1년 이상을 권장합니다.")
    split_index = max(1, int(len(unique_hours) * 0.8))
    validation_start = pd.Timestamp(unique_hours[split_index])
    train_mask = panel["timestamp"] < validation_start
    test_mask = ~train_mask
    validation_model = RandomForestRegressor(
        n_estimators=70, max_depth=14, min_samples_leaf=4, max_features=0.9, random_state=42, n_jobs=-1
    )
    validation_model.fit(x.loc[train_mask], y[train_mask.to_numpy()])
    test_predictions = validation_model.predict(x.loc[test_mask])
    focus_test = panel.loc[test_mask, "station_id"].astype(str).eq(str(focus_id)).to_numpy()
    mae = float(mean_absolute_error(y[test_mask.to_numpy(), 0][focus_test], test_predictions[focus_test, 0])) if focus_test.any() else None

    model = RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=4, max_features=0.9, random_state=42, n_jobs=-1
    )
    model.fit(x, y)
    future_start = pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1)
    future_times = pd.date_range(future_start, periods=horizon, freq="h")
    stations = panel[["station_id", "station_name"]].drop_duplicates("station_id")
    future = stations.assign(_join=1).merge(pd.DataFrame({"timestamp": future_times, "_join": 1}), on="_join").drop(columns="_join")
    future["season"] = future["timestamp"].dt.month.map(SEASON_NAMES)
    xf = feature_table(future, x_columns)
    prediction = np.maximum(0, model.predict(xf))
    future["rentals_forecast"] = prediction[:, 0]
    future["returns_forecast"] = prediction[:, 1]
    return future, panel, mae, cutoff, int(panel["station_id"].nunique())


def stock_for_name(stock: pd.DataFrame, name: str):
    if stock.empty:
        return None
    key = key_text(name)
    matches = stock[stock["station_name"].map(key_text) == key]
    return None if matches.empty else matches.iloc[0]


def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(item) for item in (lat1, lon1, lat2, lon2)):
        return None
    earth = 6371.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * earth * math.asin(math.sqrt(a))


def inventory_table(forecast: pd.DataFrame, stock: pd.DataFrame, ratio: float):
    if stock.empty:
        return pd.DataFrame()
    daily = forecast.groupby(["station_id", "station_name"], as_index=False).agg(
        expected_rentals=("rentals_forecast", "sum"), expected_returns=("returns_forecast", "sum")
    )
    items = stock.copy()
    items["name_key"] = items["station_name"].map(key_text)
    daily["name_key"] = daily["station_name"].map(key_text)
    merged = daily.merge(items.drop(columns=["station_id"], errors="ignore"), on="name_key", how="inner", suffixes=("", "_stock"))
    if merged.empty:
        return merged
    merged = merged.dropna(subset=["current_bikes", "capacity"])
    merged["target_stock"] = np.ceil(merged["capacity"] * ratio)
    # Use the lowest projected stock during the forecast window, not only the
    # final hour, so an early shortage is not hidden by later returns.
    minimum_stocks = []
    shortage_times = []
    for _, item in merged.iterrows():
        station_forecast = forecast[forecast["station_name"].map(key_text) == item["name_key"]].sort_values("timestamp")
        path = float(item["current_bikes"])
        values = []
        first_shortage = pd.NaT
        for _, hour_row in station_forecast.iterrows():
            path = float(np.clip(path + hour_row["returns_forecast"] - hour_row["rentals_forecast"], 0, item["capacity"]))
            values.append(path)
            if pd.isna(first_shortage) and path < item["target_stock"]:
                first_shortage = hour_row["timestamp"]
        minimum_stocks.append(min(values) if values else float(item["current_bikes"]))
        shortage_times.append(first_shortage)
    merged["minimum_projected_stock"] = minimum_stocks
    merged["first_shortage_time"] = shortage_times
    merged["hours_until_shortage"] = (
        (pd.to_datetime(merged["first_shortage_time"]) - pd.Timestamp.now()).dt.total_seconds() / 3600
    )
    merged["projected_stock"] = (merged["current_bikes"] + merged["expected_returns"] - merged["expected_rentals"]).clip(lower=0, upper=merged["capacity"])
    merged["shortage"] = (merged["target_stock"] - merged["minimum_projected_stock"]).clip(lower=0)
    donor_floor = np.ceil(merged["capacity"] * min(0.9, ratio + 0.15))
    merged["surplus"] = (merged["minimum_projected_stock"] - donor_floor).clip(lower=0)
    return merged


def transfer_suggestions(
    table: pd.DataFrame,
    road_factor: float,
    field_speed_kmh: float,
    citizen_speed_kmh: float,
    field_buffer_min: int,
    citizen_buffer_min: int,
):
    columns = ["보내는 대여소", "받는 대여소", "이동 참고량", "직선거리", "예상 소요", "부족 예상", "늦어도 출발", "권장 방식", "실행 제안", "추천 이유"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    donors = table[table["surplus"] >= 1].copy()
    receivers = table[table["shortage"] >= 1].copy()
    donors["left"] = donors["surplus"].astype(int)
    receivers["left"] = receivers["shortage"].astype(int)
    results = []
    for ridx, receiver in receivers.iterrows():
        while receivers.loc[ridx, "left"] > 0:
            options = donors[donors["left"] > 0]
            if options.empty:
                break
            distances = []
            for didx, donor in options.iterrows():
                dist = haversine_km(donor.get("latitude"), donor.get("longitude"), receiver.get("latitude"), receiver.get("longitude"))
                distances.append((float("inf") if dist is None else dist, didx, dist))
            _, didx, dist = min(distances, key=lambda item: item[0])
            qty = min(int(donors.loc[didx, "left"]), int(receivers.loc[ridx, "left"]))
            lead_hours = receivers.loc[ridx].get("hours_until_shortage", np.nan)
            shortage_time = receivers.loc[ridx].get("first_shortage_time", pd.NaT)
            urgent = pd.notna(lead_hours) and float(lead_hours) <= 2
            nearby = dist is not None and dist <= 1.0
            citizen_candidate = qty <= 5 and nearby and not urgent
            if citizen_candidate:
                method = "시민 참여 후보"
                action = "출발지에서 대여 → 도착지에 반납 미션 안내"
                reason = "소량·1km 이내·2시간 이상 여유"
                speed_kmh = citizen_speed_kmh
                buffer_min = citizen_buffer_min
            else:
                method = "현장팀 우선"
                action = "현장팀 차량으로 출발지 수거 → 도착지 보충"
                speed_kmh = field_speed_kmh
                buffer_min = field_buffer_min
                reasons = []
                if urgent:
                    reasons.append("부족까지 2시간 이내")
                if qty > 5:
                    reasons.append("이동량 6대 이상")
                if dist is None:
                    reasons.append("위치자료 확인 필요")
                elif not nearby:
                    reasons.append("직선거리 1km 초과")
                reason = " · ".join(reasons) if reasons else "시민 참여 조건에 해당하지 않음"
            if dist is not None and pd.notna(shortage_time):
                estimated_minutes = int(math.ceil((float(dist) * road_factor / speed_kmh) * 60 + buffer_min))
                dispatch_by = pd.Timestamp(shortage_time) - pd.Timedelta(minutes=estimated_minutes)
                now = pd.Timestamp.now()
                if dispatch_by <= now:
                dispatch_label = "출발 권고 시각 경과 · 즉시 판단"
                elif dispatch_by - now <= pd.Timedelta(minutes=5):
                    dispatch_label = "지금 출발 권고"
                else:
                    dispatch_label = dispatch_by.strftime("%m/%d %H:%M까지")
                duration_label = f"약 {estimated_minutes}분*"
            else:
                duration_label = "산정 자료 필요"
                dispatch_label = "거리·부족 시각 필요"
            results.append({
                "보내는 대여소": donors.loc[didx, "station_name"],
                "받는 대여소": receivers.loc[ridx, "station_name"],
                "이동 참고량": f"{qty}대",
                "직선거리": f"{dist:.1f} km" if dist is not None else "위치자료 없음",
                "예상 소요": duration_label,
                "부족 예상": f"약 {max(0, int(round(float(lead_hours))))}시간 후" if pd.notna(lead_hours) else "예측 기간 내",
                "최소 출발 시각": dispatch_label,
                "권장 방식": method,
                "실행 제안": action,
                "추천 이유": reason,
            })
            donors.loc[didx, "left"] -= qty
            receivers.loc[ridx, "left"] -= qty
    return pd.DataFrame(results, columns=columns)


def sidebar(history, is_demo, demo_stock):
    with st.sidebar:
        st.markdown("# 🚲 따릉이 시간대 분석")
        st.caption("시간별 대여·반납과 계절 패턴으로 재배치 참고량 계산")
        st.markdown("---")
        upload = st.file_uploader("시간대 집계 CSV", type=["csv"], help="prepare_hourly_data.py가 만든 timestamp, station_id, station_name, rentals, returns 파일")
        if upload:
            try:
                history = standardize_hourly(read_csv(upload))
                is_demo = False
                st.success(f"시간 기록 {len(history):,}행 불러옴")
            except Exception as exc:
                st.error(str(exc))
        ids = history.sort_values("timestamp").groupby("station_id")["station_name"].last().to_dict()
        station_ids = list(ids)
        focus = st.selectbox("분석 대여소", station_ids, format_func=lambda sid: ids.get(sid, str(sid)))
        horizon = st.select_slider("예측 범위", options=[1, 2, 6, 12, 24], value=24, format_func=lambda h: f"다음 {h}시간")
        ratio = st.slider("최소 재고 기준", 10, 50, 25, 5) / 100
        with st.expander("이동시간 계산 기준"):
            st.caption("직선거리에 도로 우회율을 곱한 뒤, 설정 평균속도와 준비 여유시간으로 추정합니다.")
            road_factor = st.slider("도로 우회율", 1.0, 2.0, 1.3, 0.1, help="직선거리 대비 실제 도로 경로를 가정하는 배율")
            field_speed = st.number_input("현장 차량 평균속도 (km/h)", min_value=5.0, max_value=60.0, value=20.0, step=1.0)
            citizen_speed = st.number_input("시민 자전거 평균속도 (km/h)", min_value=5.0, max_value=30.0, value=12.0, step=1.0)
            field_buffer = st.number_input("현장팀 상·하차 여유 (분)", min_value=0, max_value=60, value=15, step=5)
            citizen_buffer = st.number_input("시민 대여·반납 여유 (분)", min_value=0, max_value=30, value=5, step=5)
        st.markdown("---")
        st.markdown("**현재 재고 연결**")
        api_key = st.text_input("서울시 Open API 인증키", type="password")
        if st.button("실시간 재고 불러오기", use_container_width=True, disabled=not bool(api_key.strip())):
            try:
                with st.spinner("실시간 대여소 상태를 불러오는 중…"):
                    st.session_state["hourly_stock"] = fetch_realtime_stock(api_key.strip())
                    st.session_state["hourly_stock_label"] = "서울시 실시간 API"
                st.success(f"{len(st.session_state['hourly_stock']):,}개소 확인")
            except Exception as exc:
                st.error(str(exc))
        stock_upload = st.file_uploader("또는 현재 재고 CSV", type=["csv"], key="hourly_stock_upload")
        if stock_upload:
            try:
                st.session_state["hourly_stock"] = standardize_stock(read_csv(stock_upload))
                st.session_state["hourly_stock_label"] = "업로드한 재고 CSV"
            except Exception as exc:
                st.error(str(exc))
        stock = st.session_state.get("hourly_stock", pd.DataFrame())
        if is_demo and stock.empty:
            stock = demo_stock
        match = stock_for_name(stock, ids.get(focus, ""))
        current = st.number_input("선택 대여소 현재 자전거 수", min_value=0, max_value=500, value=int(match["current_bikes"]) if match is not None else 12, disabled=match is not None)
        capacity = st.number_input("선택 대여소 거치대 수", min_value=1, max_value=1000, value=int(match["capacity"]) if match is not None and pd.notna(match.get("capacity")) else 24, disabled=match is not None)
        st.markdown("---")
        st.caption("시간 데이터 출처")
        st.markdown(f"[대여 이력 원본 ↗]({HISTORY_URL})  ")
        st.markdown(f"[시간대별 이용 정보 ↗]({HOURLY_URL})")
    return history, is_demo, focus, horizon, ratio, road_factor, field_speed, citizen_speed, field_buffer, citizen_buffer, stock, current, capacity


def main():
    demo, demo_stock = make_demo_hourly()
    history, is_demo, focus_id, horizon, ratio, road_factor, field_speed, citizen_speed, field_buffer, citizen_buffer, stock, manual_current, manual_capacity = sidebar(demo, True, demo_stock)
    if is_demo and stock.empty:
        stock = demo_stock
        st.session_state["hourly_stock_label"] = "화면 예시 재고"
    elif "hourly_stock" in st.session_state:
        stock = st.session_state["hourly_stock"]
    if isinstance(history["timestamp"].dtype, pd.DatetimeTZDtype):
        history["timestamp"] = history["timestamp"].dt.tz_localize(None)
    names = history.sort_values("timestamp").groupby("station_id")["station_name"].last().to_dict()
    focus_name = names.get(focus_id, str(focus_id))
    label = "화면 예시 데이터" if is_demo else "서울시 시간대 기록"
    pill = "pill demo" if is_demo else "pill"

    st.markdown('<div class="hero"><div class="hero-icon">🚲</div><h1>시간대·계절 반영 따릉이 수요 예측</h1></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtle">출퇴근·심야 시간과 봄·여름·가을·겨울의 이용 패턴을 반영해 다음 {horizon}시간 대여소 재배치를 살펴봅니다. &nbsp; <span class="{pill}">{label}</span></div>', unsafe_allow_html=True)
    if is_demo:
        st.markdown('<div class="note">화면 확인용 예시 숫자입니다. 실제 분석은 서울시 대여이력을 시간 단위로 변환한 CSV를 올려야 합니다.</div>', unsafe_allow_html=True)
    else:
        start_date = pd.Timestamp(history["timestamp"].min()).date()
        end_date = pd.Timestamp(history["timestamp"].max()).date()
        days = (end_date - start_date).days + 1
        if days < 365:
            st.markdown(f'<div class="note">자료 범위: {start_date} ~ {end_date} ({days}일). 네 계절 패턴을 비교하려면 최소 1년 이상의 자료가 필요합니다.</div>', unsafe_allow_html=True)
        else:
            st.caption(f"학습 기록 범위: {start_date} ~ {end_date} · {days:,}일")

    try:
        with st.spinner("시간대·계절 예측 모델을 학습하고 있습니다…"):
            forecast, panel, mae, cutoff, station_count = train_and_forecast(history, str(focus_id), horizon)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    selected = forecast[forecast["station_id"].astype(str) == str(focus_id)].copy().sort_values("timestamp")
    if selected.empty:
        st.error("선택한 대여소 예측 결과가 없습니다. 시간 집계 자료의 대여소 번호를 확인하세요.")
        st.stop()
    current_row = stock_for_name(stock, focus_name)
    current_bikes = int(current_row["current_bikes"]) if current_row is not None else int(manual_current)
    capacity = int(current_row["capacity"]) if current_row is not None and pd.notna(current_row.get("capacity")) else int(manual_capacity)
    target = int(math.ceil(capacity * ratio))
    path = [current_bikes]
    for _, row in selected.iterrows():
        path.append(float(np.clip(path[-1] + row["returns_forecast"] - row["rentals_forecast"], 0, capacity)))
    min_index = int(np.argmin(path[1:]))
    min_bikes = int(round(path[min_index + 1]))
    risk_time = selected.iloc[min_index]["timestamp"]
    need = max(0, target - min_bikes)
    rentals_sum = int(round(selected["rentals_forecast"].sum()))
    returns_sum = int(round(selected["returns_forecast"].sum()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"다음 {horizon}시간 예상 대여", f"{rentals_sum:,}대")
    c2.metric(f"다음 {horizon}시간 예상 반납", f"{returns_sum:,}대")
    c3.metric("최저 예상 재고", f"{min_bikes}대 · {risk_time:%H시}", help="현재 재고에서 시간별 예상 반납을 더하고 대여를 뺀 값")
    c4.metric("최소 기준까지 이동 참고량", f"{need}대", help=f"거치대 {capacity}대, 최소 재고 기준 {target}대")

    moves = inventory_table(forecast, stock, ratio)
    suggestions = transfer_suggestions(moves, road_factor, field_speed, citizen_speed, field_buffer, citizen_buffer)
    st.markdown('<div class="section-title">재고 이동과 권장 실행 방식</div>', unsafe_allow_html=True)
    if is_demo:
        st.caption("예시 재고와 예시 모델 결과를 표시 중입니다.")
    if moves.empty:
        st.info("출발·도착 대여소를 찾으려면 서울시 실시간 재고 API를 연결하거나 재고 CSV를 올려주세요. 선택 대여소는 왼쪽의 현재 재고 입력값으로 계산합니다.")
    elif suggestions.empty:
        st.info("예측 기간 중, 설정한 최소 재고 기준으로 서로 옮길 부족·여유 대여소 조합이 없습니다.")
    else:
        field = suggestions[suggestions["권장 방식"] == "현장팀 우선"]
        citizen = suggestions[suggestions["권장 방식"] == "시민 참여 후보"]
        action1, action2 = st.columns(2)
        action1.metric("현장팀 우선 이동", f"{len(field)}건", help="부족 시점이 임박했거나 이동량·거리가 큰 후보")
        action1.caption("차량으로 여유 대여소에서 수거해 부족 대여소에 보충")
        action2.metric("시민 참여 후보", f"{len(citizen)}건", help="소량·근거리이고 예측상 시간 여유가 있는 후보")
        action2.caption("앱에서 여유 대여소 대여 → 부족 대여소 반납 미션 안내")
        st.dataframe(suggestions.head(8), use_container_width=True, hide_index=True)
        st.caption("시민 참여 후보는 앱 미션·보상 정책이 실제로 적용되는 대여소에서만 운영할 수 있습니다. 이 화면은 미션을 자동 발행하거나 보상을 지급하지 않습니다.")
        st.caption("* 늦어도 출발할 시각 = 부족 예상 시각 − (직선거리 × 도로 우회율 ÷ 평균속도 × 60 + 준비 여유분). 실시간 교통·실제 도로 경로는 반영하지 않는 계획용 추정치입니다.")
    st.caption(f"선택 대여소: {focus_name} · 현재 {current_bikes}대 · 거치대 {capacity}대 · 최소 기준 {target}대 · 학습 대여소 {station_count}곳")

    left, right = st.columns([1.08, 0.92], gap="medium")
    with left:
        st.markdown('<div class="section-title">시간대별 대여·반납 예측</div>', unsafe_allow_html=True)
        demand_chart = selected.set_index("timestamp")[["rentals_forecast", "returns_forecast"]].rename(columns={"rentals_forecast": "예상 대여", "returns_forecast": "예상 반납"})
        st.bar_chart(demand_chart, color=["#2e7dd1", "#249465"], height=280)
        st.caption("시간별 대여·반납 예측량 (대)")
    with right:
        st.markdown('<div class="section-title">시간대별 보유 자전거 전망</div>', unsafe_allow_html=True)
        stock_chart = pd.DataFrame(
            {"예상 보유": path, "최소 기준": [target] * len(path)},
            index=["현재"] + [ts.strftime("%H시") for ts in selected["timestamp"]],
        )
        st.line_chart(stock_chart, color=["#2e7dd1", "#e78748"], height=280)
        st.caption("현재 재고에서 시간별 대여·반납 예상량을 누적한 참고치")

    st.markdown('<div class="section-title">계절·출퇴근·심야 패턴</div>', unsafe_allow_html=True)
    selected_panel = panel[panel["station_id"].astype(str) == str(focus_id)].copy()
    selected_panel["hour"] = selected_panel["timestamp"].dt.hour
    season_hour = selected_panel.groupby(["season", "hour"])[["rentals", "returns"]].mean().reset_index()
    a, b = st.columns([1.05, 0.95], gap="medium")
    with a:
        st.caption("계절별 시간당 평균 대여 건수")
        season_chart = season_hour.pivot(index="hour", columns="season", values="rentals").reindex(columns=["봄", "여름", "가을", "겨울"])
        st.line_chart(season_chart, color=["#64a873", "#e8a442", "#d77a47", "#4f7ead"], height=250)
    with b:
        def period(hour):
            if hour in (7, 8, 9): return "출근 07–09시"
            if hour in (17, 18, 19, 20): return "퇴근 17–20시"
            if hour in (22, 23, 0, 1, 2, 3, 4, 5): return "심야 22–05시"
            if 10 <= hour <= 16: return "낮 10–16시"
            return "기타 시간"
        selected_panel["시간 구간"] = selected_panel["hour"].map(period)
        period_table = selected_panel.groupby("시간 구간")[["rentals", "returns"]].mean().rename(columns={"rentals": "시간당 평균 대여", "returns": "시간당 평균 반납"})
        period_table = period_table.reindex(["출근 07–09시", "낮 10–16시", "퇴근 17–20시", "심야 22–05시", "기타 시간"]).dropna(how="all").round(1)
        st.caption("구간별 시간당 평균 이용 건수")
        st.dataframe(period_table, use_container_width=True, height=250)

    with st.expander("현재 반영한 변수와 추가로 검증할 변수"):
        st.markdown(
            """
            **현재 예측 모델 입력:** 대여소 ID, 시각(0–23), 요일/주말, 월·계절, 출퇴근/심야 구간.  
            **재배치 계산 입력:** 현재 자전거 수, 거치대 용량, 대여소 위치(거리 계산용). 이 항목들은 수요 예측 특성과 구분됩니다.

            **추가 검증 후보:**
            - 기온·강수·풍속: [기상청 시간 관측](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36), [기상청 단기예보](https://data.kma.go.kr/data/rmt/rmtList.do?code=420&pgmNo=572)
            - 미세먼지 PM10·PM2.5: [한국환경공단 에어코리아 정보](https://www.data.go.kr/data/15073861/openapi.do)
            - 공휴일: [한국천문연구원 특일 정보](https://www.data.go.kr/dataset/15012690/openapi.do)
            - 대여소 좌표·거치대 정보: [서울시 대여소 정보](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do)

            강수 때 수요가 0에 가깝다거나 특정 온도 범위가 최적이라는 값은 미리 고정하지 않고, 데이터에서 검증해야 합니다. 토지이용·지하철 접근성·경사도는 별도 공간자료 결합, 고장 자전거·재배치 차량 일정은 운영 기록이 있어야 정확히 넣을 수 있습니다.
            """
        )

    with st.expander("재배치 방식 추천 기준과 운영 시 유의점"):
        st.markdown(
            """
            - **현장팀 우선:** 부족 예상까지 2시간 이내, 이동량 6대 이상, 직선거리 1km 초과 또는 위치 정보가 없는 경우입니다.
            - **시민 참여 후보:** 이동량 1–5대, 대여소 간 직선거리 1km 이내, 부족 예상까지 2시간보다 여유가 있는 경우입니다. 앱에서 과다 대여소 대여와 부족 대여소 반납 미션을 안내하는 방식으로 확장할 수 있습니다.
            - 이 기준은 프로젝트 시안의 우선순위 규칙이며, **직선거리는 실제 도로 이동거리와 다릅니다.** 차량 적재량·운행 경로·근무조 정보가 연결되기 전에는 현장팀 배차 지시로 해석하면 안 됩니다.
            - 마일리지·참여 가능 시간·대상 대여소는 운영기관 정책에 따라 달라질 수 있어 앱 연동 전 별도 확인이 필요합니다. 이 화면은 보상 지급을 확정하지 않습니다.
            """
        )

    st.markdown('<div class="footnote">모델: 시간대·요일·계절·대여소 특성을 학습한 Random Forest 회귀. 최근 시간순 검증 구간의 MAE는 과거 시간별 대여량 오차이며 다음 기간의 정확도를 보장하지 않습니다.</div>', unsafe_allow_html=True)
    st.markdown('<div class="footnote">재배치량과 현장팀·시민 참여 분류는 예측 재고 기반 참고안입니다. 도로 경로·차량 적재량·기상 특이사항·운영 미션 정책은 반영하지 않으며, 실제 배차나 마일리지 지급을 지시하지 않습니다.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="footnote">사용 자료: <a href="{HISTORY_URL}" target="_blank">서울시 대여이력 OA-15182</a> · <a href="{HOURLY_URL}" target="_blank">시간대별 이용정보 OA-15245</a> · <a href="{REALTIME_URL}" target="_blank">실시간 재고 OA-15493</a></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
