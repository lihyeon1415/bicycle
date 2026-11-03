from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tabareungi-hourly-preview.png"
W, H = 1500, 1360
BG, INK, MUTED, BLUE, GREEN, ORANGE, LINE = "#F4F6F9", "#1C2738", "#718096", "#2E7DD1", "#249465", "#E78748", "#E3E9F0"
im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)
regular = Path(r"C:\Windows\Fonts\malgun.ttf")
bold = Path(r"C:\Windows\Fonts\malgunbd.ttf")
if not regular.exists():
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
if not bold.exists():
    bold = regular


def ft(size, is_bold=False):
    return ImageFont.truetype(str(bold if is_bold else regular), size)


def t(x, y, value, size=14, color=INK, strong=False, anchor=None):
    d.text((x, y), value, font=ft(size, strong), fill=color, anchor=anchor)


def box(coords, radius=12, fill="white", outline=LINE, width=1):
    d.rounded_rectangle(coords, radius=radius, fill=fill, outline=outline, width=width)


def bicycle(cx, cy, scale=1):
    color = "#258358"
    r = int(6 * scale)
    left, right = (cx - 10 * scale, cy + 5 * scale), (cx + 10 * scale, cy + 5 * scale)
    width = max(1, int(2 * scale))
    d.ellipse((left[0]-r, left[1]-r, left[0]+r, left[1]+r), outline=color, width=width)
    d.ellipse((right[0]-r, right[1]-r, right[0]+r, right[1]+r), outline=color, width=width)
    d.line((left, (cx-2*scale, cy-5*scale), right, left), fill=color, width=width)
    d.line(((cx-2*scale, cy-5*scale), (cx+2*scale, cy-5*scale)), fill=color, width=width)
    d.line(((cx-2*scale, cy-5*scale), (cx-6*scale, cy-8*scale)), fill=color, width=width)


def chevron(cx, cy):
    d.line((cx-4, cy-2, cx, cy+2, cx+4, cy-2), fill="#8190A3", width=2)


# Sidebar
d.rectangle((0, 0, 270, H), fill="white")
d.line((269, 0, 269, H), fill=LINE, width=1)
box((25, 27, 69, 71), 13, "#E8F5EE")
bicycle(47, 45, .72)
t(82, 29, "따릉이 시간대 분석", 18, INK, True)
t(26, 92, "시간·계절별 수요와 재배치 추천", 11, MUTED)
t(26, 143, "1. 시간대 집계 기록", 14, INK, True)
t(26, 171, "대여소별 시간 집계 CSV", 12, MUTED)
box((24, 193, 245, 239), 9)
t(38, 207, "hourly_counts.csv 업로드", 11, "#526174")
box((26, 255, 244, 285), 8, "#FFF2D9", "#FFF2D9")
t(38, 262, "화면 예시 자료 사용 중", 11, "#956315", True)
t(26, 318, "분석 대여소", 12, MUTED)
box((24, 341, 245, 384), 9)
t(38, 353, "광화문역 7번출구", 13)
chevron(225, 363)
t(26, 409, "예측 범위", 12, MUTED)
box((24, 431, 245, 473), 9)
t(38, 443, "다음 24시간", 13)
chevron(225, 453)
t(26, 499, "최소 재고 기준", 12, MUTED)
box((24, 522, 245, 564), 9)
t(38, 535, "거치대의 25%", 13)
t(26, 605, "2. 현재 재고 연결", 14, INK, True)
t(26, 634, "서울시 Open API 인증키", 12, MUTED)
box((24, 657, 245, 700), 9)
t(38, 671, "API 키 입력 후 조회", 11, "#9AA5B4")
box((24, 715, 245, 758), 9, "#F7F9FB")
t(50, 728, "실시간 재고 불러오기", 12, "#8793A1")
box((24, 775, 245, 818), 9)
t(38, 789, "또는 현재 재고 CSV", 12, "#526174")
t(26, 856, "공식 자료", 12, MUTED, True)
t(26, 882, "대여이력 원본  ↗", 12, BLUE, True)
t(26, 908, "시간대별 이용정보  ↗", 12, BLUE, True)

# Header
x0 = 310
box((x0, 31, x0+46, 77), 13, "#E8F5EE")
bicycle(x0+23, 49, .86)
t(x0+61, 29, "시간대·계절 반영 따릉이 수요 예측", 26, INK, True)
t(x0+61, 69, "출퇴근·심야 시간과 계절별 이용 패턴으로 다음 24시간 재배치를 살펴봅니다.", 14, MUTED)
box((1288, 39, 1458, 70), 16, "#FFF2D9", "#FFF2D9")
t(1373, 49, "화면 예시 자료", 11, "#956315", True, "mm")
box((310, 111, 1458, 162), 11, "#FFF9EB", "#F2D491")
t(332, 129, "미리보기용 예시 숫자입니다. 실제 분석은 서울시 대여이력을 시간 단위로 변환한 CSV를 올려주세요.", 13, "#765819")

# KPI cards
cards = [
    (310, "다음 24시간 예상 대여", "186대", "대여소 기준 시간대 합계", BLUE),
    (598, "다음 24시간 예상 반납", "174대", "대여소 기준 시간대 합계", GREEN),
    (886, "재고 부족 예상 시각", "08시 · 0대", "선택 대여소 최저 재고", ORANGE),
    (1174, "시간별 대여 MAE", "2.1대/시간", "시간순 과거 검증 구간", INK),
]
for x, title, value, detail, color in cards:
    box((x, 184, x+270, 296), 13)
    t(x+18, 201, title, 12, MUTED)
    t(x+18, 229, value, 24, color, True)
    t(x+18, 269, detail, 11, MUTED)

# Rebalancing method recommendations
t(310, 327, "재고 이동과 권장 실행 방식", 18, INK, True)
box((310, 360, 1458, 565), 13)
box((330, 373, 880, 430), 9, "#FFF4E8", "#F4DFC8")
t(346, 380, "현장팀 우선 · 1건", 12, "#9B531D", True)
t(346, 398, "차량으로 여유 대여소에서 수거 → 부족 대여소 보충", 10, MUTED)
t(346, 416, "부족 임박 · 6대 이상 · 1km 초과 우선", 9, MUTED)
box((894, 373, 1438, 430), 9, "#EAF6EF", "#D7EBDD")
t(910, 380, "시민 참여 후보 · 1건", 12, "#21744C", True)
t(910, 398, "시민앱에서 여유 대여소 대여 → 부족 대여소 반납 안내", 10, MUTED)
t(910, 416, "5대 이하 · 1km 이내 · 2시간 이상 여유", 9, MUTED)
t(335, 438, "이동시간 가정: 도로 우회율 1.3 · 차량 20km/h · 시민 자전거 12km/h", 9, MUTED)
for x, label in [(335, "출발지 → 도착지"), (700, "수량·거리"), (850, "부족 예상"), (970, "이동 추정"), (1085, "늦어도 출발"), (1270, "권장 방식")]:
    t(x, 454, label, 9, MUTED, True)
d.line((333, 474, 1434, 474), fill=LINE, width=1)
t(335, 484, "서울역 2번출구 → 광화문역 7번출구", 10, INK, True)
t(700, 484, "8대 · 1.8km", 9)
t(850, 484, "1시간 후", 9, "#9B531D")
t(970, 484, "약 22분", 9)
t(1085, 484, "07:38까지", 9, "#9B531D")
box((1264, 480, 1400, 503), 7, "#FFF0E4", "#FFF0E4")
t(1332, 491, "현장팀 우선", 9, "#9B531D", True, "mm")
d.line((333, 511, 1434, 511), fill="#F0F3F6", width=1)
t(335, 521, "시청역 1번출구 → 광화문역 7번출구", 10, INK)
t(700, 521, "2대 · 0.4km", 9)
t(850, 521, "5시간 후", 9)
t(970, 521, "약 8분", 9)
t(1085, 521, "11:52까지", 9)
box((1264, 517, 1420, 540), 7, "#EAF6EF", "#EAF6EF")
t(1342, 528, "시민 참여 후보", 9, "#21744C", True, "mm")
t(335, 546, "추정치입니다. 도로 경로·실시간 정체를 반영하지 않으며 마일리지는 실제 연동 전 지급되지 않습니다.", 9, MUTED)

# Hourly demand + stock charts
t(310, 585, "하루 안에서 수요가 언제 달라질까요?", 18, INK, True)
box((310, 620, 865, 919), 13)
box((885, 620, 1458, 919), 13)
t(337, 640, "시간별 예상 대여 · 반납 (대)", 12, MUTED)
t(911, 640, "시간별 보유 자전거 전망", 12, MUTED)

# grouped hourly columns
lx, ly, lw, lh = 365, 694, 455, 171
for tick, fraction in [(0, 1), (10, .75), (20, .5), (30, .25), (40, 0)]:
    y = ly + lh*fraction
    d.line((lx, y, lx+lw, y), fill="#E9EDF2", width=1)
    t(lx-12, y, str(tick), 9, MUTED, anchor="rm")
rents = [2,1,1,0,0,1,3,12,19,12,8,7,9,7,6,8,12,18,22,15,9,6,4,3]
rets = [1,1,0,0,0,1,4,16,17,9,8,8,7,8,7,9,17,19,14,9,7,5,4,2]
step = lw/24
for i, (r, rr) in enumerate(zip(rents, rets)):
    bx = lx + i*step + 2
    bw = 6
    h1, h2 = r/40*lh, rr/40*lh
    d.rectangle((bx, ly+lh-h1, bx+bw, ly+lh), fill=BLUE)
    d.rectangle((bx+bw+1, ly+lh-h2, bx+2*bw+1, ly+lh), fill=GREEN)
for hour in [0, 3, 6, 9, 12, 15, 18, 21, 23]:
    t(lx+hour*step+step/2, ly+lh+12, f"{hour:02d}", 9, MUTED, anchor="mt")
box((657, 844, 668, 855), 3, BLUE, BLUE)
t(674, 841, "대여", 10, MUTED)
box((728, 844, 739, 855), 3, GREEN, GREEN)
t(745, 841, "반납", 10, MUTED)

# inventory line
rx, ry, rw, rh = 930, 694, 471, 171
for tick, fraction in [(0, 1), (10, .67), (20, .34), (30, 0)]:
    y = ry + rh*fraction
    d.line((rx, y, rx+rw, y), fill="#E9EDF2", width=1)
    t(rx-12, y, str(tick), 9, MUTED, anchor="rm")
minimum_y = ry+rh-(6/30*rh)
d.line((rx, minimum_y, rx+rw, minimum_y), fill=ORANGE, width=2)
t(rx+rw-2, minimum_y-13, "최소 기준 6대", 9, ORANGE, anchor="ra")
stock = [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 6, 5, 4, 3, 2, 1, 2, 4, 5, 4, 3]
points=[]
for i, value in enumerate(stock):
    px=rx+i*rw/24
    py=ry+rh-(value/30*rh)
    points.append((px,py))
d.line(points, fill=BLUE, width=3, joint="curve")
for px,py in points[::3]:
    d.ellipse((px-3,py-3,px+3,py+3), fill=BLUE, outline="white", width=1)
for hour in [0, 3, 6, 9, 12, 15, 18, 21, 23]:
    t(rx+hour*rw/24, ry+rh+12, f"{hour:02d}", 9, MUTED, anchor="mt")

# seasonal profile and commute table
t(310, 950, "계절·출퇴근·심야 패턴", 18, INK, True)
box((310, 984, 955, 1234), 13)
box((975, 984, 1458, 1234), 13)
t(337, 1002, "계절별 시간당 평균 대여", 12, MUTED)
sx, sy, sw, sh = 375, 1044, 520, 144
for tick, fraction in [(0, 1), (10, .5), (20, 0)]:
    y=sy+sh*fraction
    d.line((sx,y,sx+sw,y), fill="#E9EDF2", width=1)
    t(sx-12,y,str(tick),9,MUTED,anchor="rm")
season_profiles = {
    "봄": ([1,1,1,0,0,1,3,10,16,11,8,7,9,8,7,9,14,17,18,13,8,6,4,2], "#64A873"),
    "여름": ([1,1,1,0,0,1,4,12,19,13,9,9,12,10,9,11,16,22,24,18,12,8,5,3], "#E8A442"),
    "가을": ([1,1,0,0,0,1,3,11,17,12,9,8,9,8,7,10,15,19,20,14,9,6,4,2], "#D77A47"),
    "겨울": ([0,0,0,0,0,0,1,5,8,6,4,3,4,3,3,4,7,9,10,7,4,3,2,1], "#4F7EAD"),
}
for name, (values, color) in season_profiles.items():
    coords=[]
    for hour,value in enumerate(values):
        coords.append((sx+hour*sw/23, sy+sh-value/25*sh))
    d.line(coords, fill=color, width=3, joint="curve")
    d.ellipse((sx+sw+7, coords[-1][1]-3, sx+sw+13, coords[-1][1]+3), fill=color)
    # Legend at the top-right of the panel
    legend_x = 747 + list(season_profiles).index(name)*49
    box((legend_x, 966, legend_x+9, 975), 2, color, color)
    t(legend_x+13, 964, name, 9, MUTED)
for hour in [0, 6, 12, 18, 23]:
    t(sx+hour*sw/23, sy+sh+11, f"{hour:02d}시", 9, MUTED, anchor="mt")

t(1000, 1002, "시간 구간별 평균 이용 (대/시간)", 12, MUTED)
for x, label in [(1000,"시간 구간"),(1195,"대여"),(1308,"반납")]:
    t(x, 1041, label, 11, MUTED, True)
d.line((998, 1064, 1435, 1064), fill=LINE, width=1)
rows=[("출근 07–09시",14.2,13.4),("낮 10–16시",7.1,7.8),("퇴근 17–20시",18.6,15.2),("심야 22–05시",1.8,2.3)]
for idx,(label,rent,ret) in enumerate(rows):
    y=1083+idx*34
    t(1000,y,label,11,INK)
    t(1210,y,f"{rent:.1f}",11,BLUE,True)
    t(1320,y,f"{ret:.1f}",11,GREEN,True)
    if idx<3: d.line((998,y+25,1435,y+25),fill="#F0F3F6",width=1)

t(310, 1280, "※ 화면 수치는 시안용 예시입니다. 실제 계산은 시간별 대여·반납 원본과 현재 재고 자료를 연결한 뒤 표시됩니다.", 10, MUTED)
t(310, 1303, "서울시 대여이력 OA-15182 · 시간대별 이용정보 OA-15245 · 실시간 재고 OA-15493", 10, MUTED)
im.save(OUT, optimize=True)
print(OUT)
