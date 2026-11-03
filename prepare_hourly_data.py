"""Convert Seoul bike trip records into station-by-hour rental/return counts.

Example:
    python prepare_hourly_data.py --input "raw_trips_2025.csv" --output hourly_counts.csv
    python prepare_hourly_data.py --input 2025.zip 2026-01.csv --output hourly_counts.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 200_000


def clean_label(value: object) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value).strip().lower())


def find_column(columns, aliases: list[str]) -> str | None:
    normalized = {clean_label(column): column for column in columns}
    for alias in aliases:
        if clean_label(alias) in normalized:
            return normalized[clean_label(alias)]
    for key, original in normalized.items():
        if any(clean_label(alias) in key for alias in aliases if len(clean_label(alias)) >= 4):
            return original
    return None


def detect_encoding(file_obj) -> str:
    sample = file_obj.read(65536)
    file_obj.seek(0)
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp949"


def parse_datetime(values: pd.Series) -> pd.Series:
    raw = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    digits = raw.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_datetime(raw, errors="coerce")
    for length, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (10, "%Y%m%d%H"), (8, "%Y%m%d")):
        mask = digits.str.len().eq(length)
        parsed.loc[mask] = pd.to_datetime(digits.loc[mask], format=fmt, errors="coerce")
    return parsed


def csv_streams(input_path: Path):
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise ValueError(f"ZIP 안에서 CSV를 찾지 못했습니다: {input_path}")
            for member in members:
                with archive.open(member) as stream:
                    yield f"{input_path.name}:{member}", stream
    else:
        with input_path.open("rb") as stream:
            yield input_path.name, stream


def station_values(frame: pd.DataFrame, id_col: str | None, name_col: str | None) -> tuple[pd.Series, pd.Series]:
    if id_col:
        station_id = frame[id_col].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    elif name_col:
        station_id = frame[name_col].astype("string").str.strip()
    else:
        raise ValueError("대여소 ID 또는 이름 열을 찾지 못했습니다.")
    if name_col:
        station_name = frame[name_col].astype("string").str.strip()
        station_name = station_name.fillna(station_id)
    else:
        station_name = station_id
    station_id = station_id.fillna(station_name)
    return station_id, station_name


def normalize_event(frame: pd.DataFrame, time_col: str, id_col: str | None, name_col: str | None, count_col: str) -> pd.DataFrame:
    station_id, station_name = station_values(frame, id_col, name_col)
    timestamps = parse_datetime(frame[time_col])
    events = pd.DataFrame(
        {
            "timestamp": timestamps.dt.floor("h"),
            "station_id": station_id,
            "station_name": station_name,
            count_col: 1,
        }
    )
    events = events.dropna(subset=["timestamp", "station_id"])
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "station_id", "station_name", "rentals", "returns"])
    grouped = events.groupby(["timestamp", "station_id", "station_name"], as_index=False)[count_col].sum()
    if count_col == "rentals":
        grouped["returns"] = 0
    else:
        grouped["rentals"] = 0
    return grouped[["timestamp", "station_id", "station_name", "rentals", "returns"]]


def process_csv_stream(label: str, stream, connection: sqlite3.Connection) -> int:
    encoding = detect_encoding(stream)
    chunks = pd.read_csv(stream, encoding=encoding, encoding_errors="replace", dtype="string", chunksize=CHUNK_SIZE, low_memory=False)
    aliases = {
        "rent_time": ["대여일시", "대여시각", "rental_datetime", "rental_time", "start_time"],
        "rent_id": ["대여대여소번호", "대여대여소ID", "대여소번호", "start_station_id"],
        "rent_name": ["대여대여소명", "대여소명", "start_station_name"],
        "return_time": ["반납일시", "반납시각", "return_datetime", "return_time", "end_time"],
        "return_id": ["반납대여소번호", "반납대여소ID", "return_station_id"],
        "return_name": ["반납대여소명", "return_station_name"],
    }
    required = {"rent_time", "return_time"}
    total_rows = 0
    columns = None
    insert_sql = """
        INSERT INTO hourly_counts(timestamp, station_id, station_name, rentals, returns)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(timestamp, station_id, station_name) DO UPDATE SET
          rentals = hourly_counts.rentals + excluded.rentals,
          returns = hourly_counts.returns + excluded.returns
    """
    for chunk_number, chunk in enumerate(chunks, 1):
        if columns is None:
            columns = {key: find_column(chunk.columns, value) for key, value in aliases.items()}
            missing = [key for key in required if not columns[key]]
            if missing or not (columns["rent_id"] or columns["rent_name"]) or not (columns["return_id"] or columns["return_name"]):
                raise ValueError(
                    f"{label}: 대여/반납 일시와 대여소 열을 찾지 못했습니다. "
                    "서울시 대여이력 파일의 대여일시·대여대여소번호·반납일시·반납대여소번호 열을 확인하세요."
                )
        parts = []
        if columns["rent_time"]:
            parts.append(normalize_event(chunk, columns["rent_time"], columns["rent_id"], columns["rent_name"], "rentals"))
        if columns["return_time"]:
            parts.append(normalize_event(chunk, columns["return_time"], columns["return_id"], columns["return_name"], "returns"))
        if parts:
            events = pd.concat(parts, ignore_index=True)
            events = events.groupby(["timestamp", "station_id", "station_name"], as_index=False)[["rentals", "returns"]].sum()
            values = [
                (row.timestamp.strftime("%Y-%m-%d %H:%M:%S"), str(row.station_id), str(row.station_name), int(row.rentals), int(row.returns))
                for row in events.itertuples(index=False)
            ]
            connection.executemany(insert_sql, values)
            total_rows += len(chunk)
        if chunk_number % 10 == 0:
            connection.commit()
            print(f"  {label}: {total_rows:,}건 처리")
    connection.commit()
    print(f"완료: {label} ({total_rows:,}행)")
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="따릉이 대여 이력을 대여소별 시간 단위로 집계합니다.")
    parser.add_argument("--input", nargs="+", required=True, help="CSV 또는 CSV가 포함된 ZIP 경로 (여러 개 가능)")
    parser.add_argument("--output", required=True, help="집계 결과 CSV 경로")
    parser.add_argument("--top-stations", type=int, default=50, help="대여+반납 건수가 많은 대여소만 저장합니다 (기본값 50). 0이면 전체 저장")
    args = parser.parse_args()

    inputs = [Path(value) for value in args.input]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("입력 파일이 없습니다: " + ", ".join(missing))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    database = output.with_suffix(".work.sqlite")
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE hourly_counts (timestamp TEXT, station_id TEXT, station_name TEXT, rentals INTEGER, returns INTEGER, PRIMARY KEY(timestamp, station_id, station_name))"
    )
    try:
        for path in inputs:
            for label, stream in csv_streams(path):
                process_csv_stream(label, stream, connection)
        connection.commit()
        selected_ids = None
        if args.top_stations > 0:
            selected_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT station_id FROM hourly_counts GROUP BY station_id ORDER BY SUM(rentals + returns) DESC LIMIT ?",
                    (args.top_stations,),
                ).fetchall()
            ]
            connection.execute("CREATE TEMP TABLE selected_stations (station_id TEXT PRIMARY KEY)")
            connection.executemany("INSERT INTO selected_stations(station_id) VALUES (?)", [(station_id,) for station_id in selected_ids])
        query = "SELECT timestamp, station_id, station_name, rentals, returns FROM hourly_counts ORDER BY station_id, timestamp"
        if selected_ids is not None:
            query = "SELECT h.timestamp, h.station_id, h.station_name, h.rentals, h.returns FROM hourly_counts h JOIN selected_stations s ON h.station_id=s.station_id ORDER BY h.station_id, h.timestamp"
        with output.open("w", newline="", encoding="utf-8-sig") as destination:
            writer = csv.writer(destination)
            writer.writerow(["timestamp", "station_id", "station_name", "rentals", "returns"])
            cursor = connection.execute(query)
            while rows := cursor.fetchmany(100_000):
                writer.writerows(rows)
        count = connection.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()[0]
        scope = f"상위 {len(selected_ids)}개 대여소" if selected_ids is not None else "전체 대여소"
        print(f"시간대 집계 완료: {scope}, {count:,}행 → {output.resolve()}")
    finally:
        connection.close()
        if database.exists():
            database.unlink()


if __name__ == "__main__":
    main()
