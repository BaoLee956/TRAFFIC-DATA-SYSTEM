"""
================================================================================
HỆ THỐNG PHÂN TÍCH VÀ KHAI PHÁ DỮ LIỆU VI PHẠM GIAO THÔNG ĐÔ THỊ
Production-ready Streamlit Application for Spatio-Temporal Traffic Analytics
Enhanced with 6 Academic Visualization Modules
Author: Senior Data Scientist & Streamlit Expert
================================================================================
"""

from typing import Tuple, Dict, Any, List
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Optional graphviz library for object conversion (Streamlit also parses DOT strings natively)
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

# Geospatial libraries (with graceful fallback if not yet installed)
try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# ==============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="Urban Traffic Violation Analytics | DSS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic aesthetic enhancements
st.markdown("""
<style>
    /* Metric Card Enhancements */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #1E3A8A;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        color: #475569;
    }
    
    /* Callout & Card containers */
    .theory-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .hypothesis-card {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .academic-box {
        background-color: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #334155;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. DATA GENERATING PROCESS (DGP) & CACHED INGESTION
# ==============================================================================
GPS_BOUNDS = {
    "lat_min": 20.85,
    "lat_max": 21.25,
    "lon_min": 105.65,
    "lon_max": 106.05
}

HOTSPOTS_CONFIG = [
    {
        "name": "Nút giao Ngã Tư Sở (Thanh Xuân)",
        "lat": 21.0016, "lon": 105.8193, "weight": 0.26,
        "speed_limit": 50, "dominant_violation": "Đi sai làn đường"
    },
    {
        "name": "Trục Võ Chí Công - Cầu Nhật Tân",
        "lat": 21.0827, "lon": 105.8123, "weight": 0.22,
        "speed_limit": 80, "dominant_violation": "Chạy quá tốc độ"
    },
    {
        "name": "Nút Bến xe Giáp Bát - Giải Phóng",
        "lat": 20.9798, "lon": 105.8432, "weight": 0.18,
        "speed_limit": 60, "dominant_violation": "Đi sai làn đường"
    },
    {
        "name": "Nút Cầu Giấy - Kim Mã",
        "lat": 21.0315, "lon": 105.8018, "weight": 0.16,
        "speed_limit": 50, "dominant_violation": "Vượt đèn đỏ"
    },
    {
        "name": "Khu vực Phố Cổ - Hoàn Kiếm",
        "lat": 21.0313, "lon": 105.8524, "weight": 0.18,
        "speed_limit": 40, "dominant_violation": "Vi phạm nồng độ cồn"
    }
]


@st.cache_data(show_spinner=False)
def generate_mock_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Sinh dữ liệu mô phỏng theo mô hình Data Generating Process (DGP) thực tế:
    - Không gian: Mô hình hỗn hợp Gauss (Gaussian Mixture) quanh 5 nút giao trọng điểm.
    - Thời gian: Quá trình Poisson không thuần nhất với 2 đỉnh cao điểm (8h, 18h) và đỉnh cồn (22h).
    - Tốc độ: Phân phối chuẩn có điều kiện theo loại phương tiện và biển báo tốc độ.
    - Tiêm ngoại lai thực tế: GPS drift (0.0, 0.0), tốc độ âm/siêu thanh, và giá trị NaN.
    """
    np.random.seed(seed)
    
    weights = [h["weight"] for h in HOTSPOTS_CONFIG]
    weights = np.array(weights) / sum(weights)
    cluster_indices = np.random.choice(len(HOTSPOTS_CONFIG), size=n_samples, p=weights)
    
    base_lats = np.array([HOTSPOTS_CONFIG[i]["lat"] for i in cluster_indices])
    base_lons = np.array([HOTSPOTS_CONFIG[i]["lon"] for i in cluster_indices])
    speed_limits = np.array([HOTSPOTS_CONFIG[i]["speed_limit"] for i in cluster_indices])
    locations = [HOTSPOTS_CONFIG[i]["name"] for i in cluster_indices]
    
    lats = base_lats + np.random.normal(0, 0.0035, n_samples)
    lons = base_lons + np.random.normal(0, 0.0035, n_samples)
    
    # Inhomogeneous Poisson arrival rates
    peak_morning = np.random.normal(8.0, 1.1, int(n_samples * 0.35))
    peak_evening = np.random.normal(18.0, 1.3, int(n_samples * 0.38))
    peak_night = np.random.normal(22.0, 0.9, int(n_samples * 0.15))
    off_peak = np.random.uniform(0, 24, n_samples - len(peak_morning) - len(peak_evening) - len(peak_night))
    
    hours_raw = np.concatenate([peak_morning, peak_evening, peak_night, off_peak])
    np.random.shuffle(hours_raw)
    hours = np.clip(hours_raw.astype(int), 0, 23)
    
    end_date = pd.Timestamp.now().floor("D")
    start_date = end_date - pd.Timedelta(days=30)
    days_offset = np.random.randint(0, 30, size=n_samples)
    minutes = np.random.randint(0, 60, size=n_samples)
    seconds = np.random.randint(0, 60, size=n_samples)
    
    timestamps = [
        start_date + pd.Timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s))
        for d, h, m, s in zip(days_offset, hours, minutes, seconds)
    ]
    
    weather_types = ["Nắng ráo", "Mưa rào", "Sương mù / Đêm"]
    weather = np.random.choice(weather_types, size=n_samples, p=[0.60, 0.25, 0.15])
    
    vehicle_types = ["Xe máy", "Ô tô con", "Xe tải", "Xe khách"]
    vehicles = np.random.choice(vehicle_types, size=n_samples, p=[0.62, 0.24, 0.09, 0.05])
    
    all_violations = [
        "Chạy quá tốc độ", "Vượt đèn đỏ", "Đi sai làn đường",
        "Không đội mũ bảo hiểm", "Đi ngược chiều", "Vi phạm nồng độ cồn"
    ]
    
    violation_records = []
    recorded_speeds = []
    
    for i in range(n_samples):
        v = vehicles[i]
        w = weather[i]
        hr = hours[i]
        lim = speed_limits[i]
        
        if w == "Mưa rào":
            probs = [0.12, 0.38, 0.32, 0.08, 0.08, 0.02]
        elif hr >= 21 or hr <= 3:
            probs = [0.32, 0.08, 0.12, 0.08, 0.04, 0.36]
        elif v == "Xe máy":
            probs = [0.18, 0.26, 0.16, 0.26, 0.10, 0.04]
        else:
            probs = [0.46, 0.20, 0.26, 0.00, 0.04, 0.04]
            
        probs = np.array(probs) / sum(probs)
        viol = np.random.choice(all_violations, p=probs)
        violation_records.append(viol)
        
        if viol == "Chạy quá tốc độ":
            delta_mu = 18.5 if v in ["Ô tô con", "Xe tải"] else 12.0
            spd = lim + np.abs(np.random.normal(delta_mu, 6.5))
        else:
            spd = np.random.normal(lim - 6.0, 7.5)
            spd = max(15.0, min(lim + 3.0, spd))
            
        recorded_speeds.append(round(spd, 1))
        
    df = pd.DataFrame({
        "ma_vi_pham": [f"VP-HN-{240000 + i}" for i in range(n_samples)],
        "thoi_gian": timestamps,
        "vi_do": lats,
        "kinh_do": lons,
        "loai_xe": vehicles,
        "thoi_tiet": weather,
        "toc_do_ghi_nhan": recorded_speeds,
        "toc_do_gioi_han": speed_limits,
        "loai_vi_pham": violation_records,
        "dia_diem": locations
    })
    
    # Tiêm ngoại lai thực tế
    n_gps_err = int(n_samples * 0.016)
    gps_err_idx = np.random.choice(n_samples, size=n_gps_err, replace=False)
    df.loc[gps_err_idx[:n_gps_err//2], ["vi_do", "kinh_do"]] = [0.0, 0.0]
    df.loc[gps_err_idx[n_gps_err//2:], ["vi_do", "kinh_do"]] = [24.85, 102.15]
    
    n_spd_err = int(n_samples * 0.012)
    spd_err_idx = np.random.choice(n_samples, size=n_spd_err, replace=False)
    df.loc[spd_err_idx[:n_spd_err//2], "toc_do_ghi_nhan"] = -15.0
    df.loc[spd_err_idx[n_spd_err//2:], "toc_do_ghi_nhan"] = 285.5
    
    n_nan = int(n_samples * 0.01)
    nan_idx = np.random.choice(n_samples, size=n_nan, replace=False)
    df.loc[nan_idx, "thoi_tiet"] = np.nan
    
    return df


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    """Tải dữ liệu thô: đọc file local hoặc tự động khởi tạo theo DGP."""
    try:
        df = pd.read_csv("traffic_violations.csv")
        df["thoi_gian"] = pd.to_datetime(df["thoi_gian"])
        return df
    except Exception:
        return generate_mock_data(n_samples=5000, seed=42)


@st.cache_data(show_spinner=False)
def clean_traffic_pipeline(df: pd.DataFrame, apply_filters: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Pipeline tiền xử lý và làm sạch:
    1. Bounding box GPS trong phạm vi đô thị.
    2. Lọc bỏ tốc độ phi thực tế (<= 0 hoặc > 160 km/h).
    3. Xử lý thiếu dữ liệu.
    4. Trích xuất đặc trưng (giờ, cao điểm, cuối tuần, độ vượt tốc độ).
    """
    metadata = {
        "raw_count": len(df),
        "gps_dropped": 0,
        "speed_dropped": 0,
        "na_dropped": 0,
        "cleaned_count": len(df)
    }
    
    if not apply_filters:
        df_uncleaned = df.copy()
        df_uncleaned["thoi_gian"] = pd.to_datetime(df_uncleaned["thoi_gian"])
        df_uncleaned["gio"] = df_uncleaned["thoi_gian"].dt.hour
        df_uncleaned["toc_do_vuot_muc"] = (df_uncleaned["toc_do_ghi_nhan"] - df_uncleaned["toc_do_gioi_han"]).clip(lower=0)
        return df_uncleaned, metadata

    valid_lat = df["vi_do"].between(GPS_BOUNDS["lat_min"], GPS_BOUNDS["lat_max"])
    valid_lon = df["kinh_do"].between(GPS_BOUNDS["lon_min"], GPS_BOUNDS["lon_max"])
    valid_gps = valid_lat & valid_lon
    metadata["gps_dropped"] = int((~valid_gps).sum())
    
    valid_speed = df["toc_do_ghi_nhan"].between(5.0, 160.0)
    metadata["speed_dropped"] = int((~valid_speed).sum())
    
    clean_mask = valid_gps & valid_speed
    df_clean = df[clean_mask].dropna(subset=["loai_xe", "thoi_tiet", "loai_vi_pham"]).copy()
    metadata["na_dropped"] = int(len(df[clean_mask]) - len(df_clean))
    
    df_clean["thoi_gian"] = pd.to_datetime(df_clean["thoi_gian"])
    df_clean["gio"] = df_clean["thoi_gian"].dt.hour
    df_clean["ngay_trong_tuan"] = df_clean["thoi_gian"].dt.day_name()
    df_clean["la_cuoi_tuan"] = df_clean["thoi_gian"].dt.weekday >= 5
    df_clean["la_gio_cao_diem"] = df_clean["gio"].isin([7, 8, 17, 18])
    df_clean["toc_do_vuot_muc"] = (df_clean["toc_do_ghi_nhan"] - df_clean["toc_do_gioi_han"]).clip(lower=0)
    
    def categorize_time_of_day(hr: int) -> str:
        if 6 <= hr <= 9:
            return "Cao điểm Sáng (06-09h)"
        elif 10 <= hr <= 15:
            return "Ban ngày / Trưa (10-15h)"
        elif 16 <= hr <= 19:
            return "Cao điểm Chiều (16-19h)"
        else:
            return "Ban đêm (20-05h)"
            
    df_clean["khung_gio_nhom"] = df_clean["gio"].apply(categorize_time_of_day)
    metadata["cleaned_count"] = len(df_clean)
    return df_clean, metadata


# ==============================================================================
# 3. STATISTICAL & ML ENGINE (CACHED HEAVY COMPUTATIONS)
# ==============================================================================
EARTH_RADIUS_METERS = 6371008.8

@st.cache_data(show_spinner=False)
def execute_dbscan_clustering(
    coords_deg: np.ndarray,
    eps_meters: float = 400.0,
    min_samples: int = 15
) -> np.ndarray:
    """Phân cụm DBSCAN sử dụng khoảng cách mặt cầu Haversine."""
    if len(coords_deg) == 0:
        return np.array([])
        
    coords_rad = np.radians(coords_deg)
    eps_rad = eps_meters / EARTH_RADIUS_METERS
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    labels = db.fit_predict(coords_rad)
    return labels


@st.cache_data(show_spinner=False)
def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Tính toán Khoảng tin cậy Bootstrap 95% cho giá trị trung bình:
    Trả về: (mean_val, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0
        
    np.random.seed(seed)
    boot_samples = np.random.choice(data, size=(n_bootstraps, len(data)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    
    mean_val = float(np.mean(data))
    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return mean_val, ci_lower, ci_upper


@st.cache_data(show_spinner=False)
def compute_dbscan_silhouette_curve(
    coords_deg: np.ndarray,
    min_samples: int = 15,
    sample_size: int = 1200
) -> pd.DataFrame:
    """
    Tính toán đường cong Silhouette Score qua dải bán kính Epsilon để đánh giá độ gãy thuật toán:
    Tối ưu hóa: Lấy mẫu đại diện và chuyển đổi sang Radian một lần.
    """
    if len(coords_deg) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(coords_deg), size=sample_size, replace=False)
        sample_coords = coords_deg[idx]
    else:
        sample_coords = coords_deg
        
    coords_rad = np.radians(sample_coords)
    eps_range = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    results = []
    for eps in eps_range:
        eps_rad = eps / EARTH_RADIUS_METERS
        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
        labels = db.fit_predict(coords_rad)
        
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # Silhouette chỉ xác định khi có ít nhất 2 cụm hợp lệ
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters and len(set(labels[mask])) >= 2:
                score = silhouette_score(coords_rad[mask], labels[mask], metric="haversine")
            else:
                score = 0.0
        else:
            score = 0.0
            
        noise_ratio = (labels == -1).sum() / len(labels) * 100
        results.append({
            "eps_meters": eps,
            "silhouette": round(score, 3),
            "n_clusters": n_clusters,
            "noise_pct": round(noise_ratio, 1)
        })
        
    return pd.DataFrame(results)


# ==============================================================================
# 4. VIEW PAGE 1: OVERVIEW & THEORY (CƠ SỞ LÝ THUYẾT & DGP)
# ==============================================================================
def render_page_overview_theory():
    st.title("📚 Cơ sở Lý thuyết & Mô hình Sinh Dữ liệu (DGP)")
    st.caption("Khung phân tích Không - Thời gian và Kiểm định Thống kê suy diễn trong An toàn Giao thông")
    
    st.markdown("""
    <div class="theory-card">
        <h4 style="margin-top:0; color:#1E3A8A;">🎯 Bối cảnh Bài toán & Mục tiêu Khoa học</h4>
        <p>Hệ thống Giao thông Thông minh (<strong>Intelligent Transport Systems - ITS</strong>) đòi hỏi các thuật toán giám sát 
        tự động có khả năng phân tách giữa các điểm tập trung vi phạm mang tính hệ thống (<strong>Hotspots / Điểm đen</strong>) 
        và các hiện tượng vi phạm ngẫu nhiên phân tán. Thay vì chỉ áp dụng thống kê mô tả, hệ thống này thiết lập quy trình khoa học: 
        mô hình hóa <strong>Data Generating Process (DGP)</strong>, tiền xử lý dữ liệu viễn thám GPS, phân cụm mật độ không gian bằng 
        <strong>DBSCAN với metric Haversine</strong>, và kiểm định giả thuyết thống kê suy diễn chuẩn mực.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📐 1. Mô hình Toán học: Quá trình Sinh Dữ liệu (DGP)")
        st.markdown("**A. Quá trình Đến Thời gian (Inhomogeneous Poisson Process):**")
        st.latex(r"P(N(t + \Delta t) - N(t) = k) = \frac{\left(\int_t^{t+\Delta t} \lambda(u) du\right)^k e^{-\int_t^{t+\Delta t} \lambda(u) du}}{k!}")
        st.caption("Trong đó $\\lambda(t)$ đạt cực đại tại 2 khung giờ cao điểm (sáng: 07h30 - 08h30, chiều: 17h30 - 18h30) và giờ kiểm soát cồn ban đêm.")

        st.markdown("**B. Khoảng cách Cầu Haversine trên Geoid:**")
        st.latex(r"d_{\text{haversine}} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)")

    with col2:
        st.subheader("🔬 2. Thuật toán Phân cụm & Kiểm định Thống kê")
        st.markdown("**A. Phân cụm Mật độ DBSCAN:**")
        st.latex(r"N_\epsilon(p) = \{q \in \mathcal{D} \mid d_{\text{haversine}}(p, q) \le \epsilon\}, \quad |N_\epsilon(p)| \ge \text{MinPts}")
        st.caption("Các điểm không thuộc bất kỳ lân cận lõi nào được gắn nhãn $C = -1$ (Spatial Noise / Nhiễu ngẫu nhiên).")
        
        st.markdown("**B. Kiểm định Tính Độc lập Chi-Square ($\\chi^2$):**")
        st.latex(r"\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}")
        
        st.markdown("**C. Kiểm định Two-Sample Welch's T-Test:**")
        st.latex(r"t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}, \quad \nu \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}")

    st.markdown("---")
    
    st.subheader("💡 3. Các Giả thuyết Nghiên cứu (Research Hypotheses)")
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown("""
        <div class="hypothesis-card">
            <h5 style="color:#16A34A; margin-top:0;">Giả thuyết H1 (Spatial Clustering)</h5>
            <p><strong>H₀:</strong> Các vi phạm phân bố ngẫu nhiên hoàn toàn (Complete Spatial Randomness - CSR) trong không gian đô thị.</p>
            <p><strong>H₁:</strong> Vi phạm tập trung tạo thành các điểm đen mật độ cao (DBSCAN Clusters) gắn liền với cấu trúc nút giao thông trọng điểm.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("""
        <div class="hypothesis-card">
            <h5 style="color:#16A34A; margin-top:0;">Giả thuyết H2 (Weather Association)</h5>
            <p><strong>H₀:</strong> Hành vi vi phạm hoàn toàn độc lập với điều kiện thời tiết (Nắng, Mưa, Sương mù).</p>
            <p><strong>H₁:</strong> Tồn tại mối liên hệ phụ thuộc có ý nghĩa thống kê giữa thời tiết xấu và sự gia tăng hành vi vi phạm (χ²-test, p &lt; 0.05).</p>
        </div>
        """, unsafe_allow_html=True)
    with col_h3:
        st.markdown("""
        <div class="hypothesis-card">
            <h5 style="color:#16A34A; margin-top:0;">Giả thuyết H3 (Vehicle Speed Disparity)</h5>
            <p><strong>H₀:</strong> Tốc độ vượt giới hạn trung bình (Speed Excess) của Ô tô con và Xe máy là bằng nhau (μ₁ = μ₂).</p>
            <p><strong>H₁:</strong> Người điều khiển Ô tô con có mức độ vượt tốc độ vi phạm trung bình cao hơn Xe máy có ý nghĩa thống kê (Welch's t-test, p &lt; 0.05).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.subheader("📖 4. Từ điển Dữ liệu Chuẩn (Data Dictionary)")
    data_dict = pd.DataFrame([
        {"Tên Trường": "ma_vi_pham", "Kiểu": "String", "Mô tả": "Mã định danh duy nhất của biên bản vi phạm", "Giá trị Mẫu": "VP-HN-240102", "Đơn vị": "-"},
        {"Tên Trường": "thoi_gian", "Kiểu": "Datetime", "Mô tả": "Thời điểm camera / cảm biến ghi nhận hành vi", "Giá trị Mẫu": "2024-10-15 08:24:10", "Đơn vị": "ISO 8601"},
        {"Tên Trường": "vi_do", "Kiểu": "Float", "Mô tả": "Vĩ độ WGS84 đo từ GPS / Camera ITS", "Giá trị Mẫu": "21.00164", "Đơn vị": "Độ thập phân (°N)"},
        {"Tên Trường": "kinh_do", "Kiểu": "Float", "Mô tả": "Kinh độ WGS84 đo từ GPS / Camera ITS", "Giá trị Mẫu": "105.81932", "Đơn vị": "Độ thập phân (°E)"},
        {"Tên Trường": "loai_xe", "Kiểu": "Categorical", "Mô tả": "Phân loại phương tiện tham gia giao thông", "Giá trị Mẫu": "Xe máy / Ô tô con / Xe tải / Xe khách", "Đơn vị": "-"},
        {"Tên Trường": "thoi_tiet", "Kiểu": "Categorical", "Mô tả": "Điều kiện khí tượng tại thời điểm ghi nhận", "Giá trị Mẫu": "Nắng ráo / Mưa rào / Sương mù", "Đơn vị": "-"},
        {"Tên Trường": "toc_do_ghi_nhan", "Kiểu": "Float", "Mô tả": "Tốc độ tức thời đo bằng Radar / Doppler LiDAR", "Giá trị Mẫu": "68.5", "Đơn vị": "km/h"},
        {"Tên Trường": "toc_do_gioi_han", "Kiểu": "Integer", "Mô tả": "Biển báo giới hạn tốc độ tối đa quy định", "Giá trị Mẫu": "50", "Đơn vị": "km/h"},
        {"Tên Trường": "toc_do_vuot_muc", "Kiểu": "Float", "Mô tả": "Hiệu số vượt tốc độ: max(0, v_ghi_nhan - v_gioi_han)", "Giá trị Mẫu": "18.5", "Đơn vị": "km/h (Biến phái sinh)"},
        {"Tên Trường": "loai_vi_pham", "Kiểu": "Categorical", "Mô tả": "Hành vi vi phạm quy chuẩn giao thông đường bộ", "Giá trị Mẫu": "Chạy quá tốc độ / Vượt đèn đỏ", "Đơn vị": "-"},
        {"Tên Trường": "dia_diem", "Kiểu": "String", "Mô tả": "Tên trục đường / nút giao trọng điểm", "Giá trị Mẫu": "Nút giao Ngã Tư Sở", "Đơn vị": "-"}
    ])
    st.dataframe(data_dict, use_container_width=True, hide_index=True)


# ==============================================================================
# 5. VIEW PAGE 2: DATA PIPELINE & TRANSPARENCY (MODULE 1: GRAPHVIZ FLOWCHART)
# ==============================================================================
def render_page_data_pipeline(df_raw: pd.DataFrame):
    st.title("⚙️ Đường ống Tiền xử lý Dữ liệu & Tính Minh bạch (Data Pipeline)")
    st.caption("Kiểm định chất lượng dữ liệu (Data Quality Assurance) và trực quan hóa luồng lọc dữ liệu động")

    tab_raw, tab_clean = st.tabs(["📥 Raw Data (Dữ liệu Thô)", "✨ Cleaned Data (Dữ liệu Đã Tiền Xử Lý)"])

    # ------------------ TAB 1: RAW DATA ------------------
    with tab_raw:
        st.subheader("1. Kiểm tra Tổng quan Dữ liệu Thô từ Cảm biến")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        n_raw = len(df_raw)
        n_na = df_raw.isna().sum().sum()
        
        invalid_gps = ~(
            df_raw["vi_do"].between(GPS_BOUNDS["lat_min"], GPS_BOUNDS["lat_max"]) &
            df_raw["kinh_do"].between(GPS_BOUNDS["lon_min"], GPS_BOUNDS["lon_max"])
        )
        n_gps_anom = int(invalid_gps.sum())
        
        invalid_spd = ~(df_raw["toc_do_ghi_nhan"].between(5.0, 160.0))
        n_spd_anom = int(invalid_spd.sum())

        col_m1.metric("Tổng bản ghi gốc", f"{n_raw:,}")
        col_m2.metric("GPS Drift / Lạc tọa độ", f"{n_gps_anom} ({n_gps_anom/n_raw*100:.1f}%)")
        col_m3.metric("Lỗi cảm biến Tốc độ", f"{n_spd_anom} ({n_spd_anom/n_raw*100:.1f}%)")
        col_m4.metric("Ô giá trị Khuyết (NaN)", f"{n_na}")

        st.warning("""
        ⚠️ **Cảnh báo Nhiễu Dữ liệu Cảm biến:**
        - Cảm biến GPS ghi nhận các tọa độ bất thường (ví dụ: vĩ độ = 0.0, kinh độ = 0.0 do mất tín hiệu vệ tinh).
        - Cảm biến tốc độ Doppler gặp xung nhiễu điện từ, ghi nhận tốc độ âm (ví dụ: -15 km/h) hoặc siêu âm (> 250 km/h).
        """)

        st.markdown("**Bảng mẫu 10 bản ghi dữ liệu thô:**")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.download_button(
            label="💾 Tải xuống File Dữ liệu Thô (traffic_violations_raw.csv)",
            data=df_raw.to_csv(index=False).encode("utf-8-sig"),
            file_name="traffic_violations_raw.csv",
            mime="text/csv",
            help="Tải toàn bộ 5,000 bản ghi dữ liệu vi phạm thô dạng CSV"
        )

        st.markdown("**Bản ghi bất thường tiêu biểu phát hiện được:**")
        anomalies_sample = df_raw[invalid_gps | invalid_spd].head(6)
        st.dataframe(anomalies_sample, use_container_width=True)

    # ------------------ TAB 2: CLEANED DATA ------------------
    with tab_clean:
        st.subheader("2. Áp dụng Thuật toán Lọc Nhiễu & Kỹ thuật Đặc trưng (Feature Engineering)")
        
        apply_filter = st.toggle("🛡️ Áp dụng lọc nhiễu GPS & Outliers", value=True, help="Bật để kích hoạt bộ lọc chuẩn khoa học dữ liệu")

        df_cleaned, meta = clean_traffic_pipeline(df_raw, apply_filters=apply_filter)

        # ------------------ MODULE 1: DATA PIPELINE FLOWCHART (GRAPHVIZ) ------------------
        st.markdown("### 🔀 Sơ đồ Luồng Tiền xử lý Dữ liệu Động (Data Pipeline Flowchart)")
        
        # Build dynamic DOT graph string matching exact record counts
        dot_pipeline = f"""
        digraph DataPipeline {{
            rankdir=LR;
            bgcolor="transparent";
            node [fontname="Arial", shape=box, style="filled,rounded", fontsize=11, penwidth=1.5];
            edge [fontname="Arial", fontsize=10, color="#475569", penwidth=1.6];
            
            raw [label="📥 Dữ liệu Thô Ban đầu\\n(N = {meta['raw_count']:,} dòng)", fillcolor="#EFF6FF", color="#2563EB", fontcolor="#1E3A8A"];
            gps [label="📍 Lọc Nhiễu GPS Bounding Box\\n(-{meta['gps_dropped']} dòng ngoài Hà Nội)", fillcolor="#FEF2F2", color="#DC2626", fontcolor="#991B1B"];
            speed [label="⚡ Lọc Dị biệt Tốc độ Cảm biến\\n(-{meta['speed_dropped']} dòng vô lý: <=0 hoặc >160)", fillcolor="#FFFBEB", color="#D97706", fontcolor="#92400E"];
            nan [label="🧩 Xử lý Khuyết thiếu NaN\\n(-{meta['na_dropped']} dòng)", fillcolor="#F5F3FF", color="#7C3AED", fontcolor="#5B21B6"];
            clean [label="✨ Dữ liệu Sạch Chuẩn hóa\\n(Z = {meta['cleaned_count']:,} dòng | {meta['cleaned_count']/meta['raw_count']*100:.2f}%)", fillcolor="#F0FDF4", color="#16A34A", fontcolor="#166534"];
            
            raw -> gps [label=" [20.85, 21.25]°N\\n[105.65, 106.05]°E"];
            gps -> speed [label=" [5, 160] km/h"];
            speed -> nan [label=" Listwise Drop"];
            nan -> clean [label=" Feature Eng\\n(Δv, Hour, Rush)"];
        }}
        """
        st.graphviz_chart(dot_pipeline, use_container_width=True)
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {meta['raw_count']:,} bản ghi")

        if apply_filter:
            st.success(f"✅ Bộ lọc đang HOẠT ĐỘNG: Đã loại bỏ **{meta['gps_dropped']}** tọa độ GPS lạc giới hạn, **{meta['speed_dropped']}** dị biệt tốc độ phi vật lý, và **{meta['na_dropped']}** dòng khuyết thiếu.")
            
            c_ret1, c_ret2, c_ret3 = st.columns(3)
            retention_rate = (meta["cleaned_count"] / meta["raw_count"]) * 100
            c_ret1.metric("Bản ghi Hợp lệ Giữ lại", f"{meta['cleaned_count']:,}")
            c_ret2.metric("Tỷ lệ Giữ lại (Retention Rate)", f"{retention_rate:.2f}%")
            c_ret3.metric("Số Trường đặc trưng sau Engineering", f"{df_cleaned.shape[1]} cột")

            st.markdown("### 📜 Minh bạch Mã nguồn Xử lý (Data Cleaning Code Snippet):")
            st.code("""
# ------------------------------------------------------------------------------
# PIPELINE TIỀN XỬ LÝ & LỌC NGOẠI LAI CHUẨN KHOA HỌC DỮ LIỆU
# ------------------------------------------------------------------------------
def clean_traffic_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    # Bước 1: Lọc bounding box địa lý hợp lệ (Khu vực Đô thị Hà Nội)
    valid_gps = (
        df["vi_do"].between(20.85, 21.25) & 
        df["kinh_do"].between(105.65, 106.05)
    )
    
    # Bước 2: Lọc tốc độ vật lý hợp lý của phương tiện giao thông (5 km/h <= v <= 160 km/h)
    valid_speed = df["toc_do_ghi_nhan"].between(5.0, 160.0)
    
    # Bước 3: Lọc dữ liệu kết hợp và loại bỏ Missing Values
    clean_mask = valid_gps & valid_speed
    df_clean = df[clean_mask].dropna(subset=["loai_xe", "thoi_tiet", "loai_vi_pham"]).copy()
    
    # Bước 4: Feature Engineering
    df_clean["thoi_gian"] = pd.to_datetime(df_clean["thoi_gian"])
    df_clean["gio"] = df_clean["thoi_gian"].dt.hour
    df_clean["la_cuoi_tuan"] = df_clean["thoi_gian"].dt.weekday >= 5
    df_clean["la_gio_cao_diem"] = df_clean["gio"].isin([7, 8, 17, 18])
    df_clean["toc_do_vuot_muc"] = (df_clean["toc_do_ghi_nhan"] - df_clean["toc_do_gioi_han"]).clip(lower=0)
    
    return df_clean
            """, language="python")

            st.markdown("**Bảng dữ liệu sạch sau xử lý:**")
            st.dataframe(df_cleaned.head(10), use_container_width=True)
            
            st.download_button(
                label="💾 Tải xuống File Dữ liệu Đã Làm Sạch (traffic_violations_cleaned.csv)",
                data=df_cleaned.to_csv(index=False).encode("utf-8-sig"),
                file_name="traffic_violations_cleaned.csv",
                mime="text/csv",
                help="Tải toàn bộ bản ghi dữ liệu đã lọc nhiễu dạng CSV"
            )

            st.markdown("**Thống kê mô tả các biến định lượng sau làm sạch:**")
            st.dataframe(
                df_cleaned[["toc_do_ghi_nhan", "toc_do_gioi_han", "toc_do_vuot_muc", "gio"]].describe().T.round(2),
                use_container_width=True
            )
        else:
            st.error("⚠️ Bộ lọc đang TẮT! Dữ liệu đang chứa toàn bộ ngoại lai và nhiễu cảm biến.")
            st.dataframe(df_cleaned.head(10), use_container_width=True)


# ==============================================================================
# 6. VIEW PAGE 3: SPATIAL EDA & VISUALIZATIONS (MODULE 2, 3, 4, 5)
# ==============================================================================
def render_page_spatial_eda(df_clean: pd.DataFrame):
    st.title("🗺️ Khai phá Không gian & Bản đồ Điểm đen (Spatial EDA & Map)")
    st.caption("Phân tích phân phối tốc độ, kiểm định sai số bootstrap, tương quan phi tham số và phân cụm DBSCAN")

    # ------------------ GLOBAL FILTERS ------------------
    with st.expander("🔍 BỘ LỌC TỔNG THỂ (GLOBAL FILTERS)", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            hour_range = st.slider("Khung giờ vi phạm trong ngày:", 0, 23, (0, 23), step=1)
        with col_f2:
            weather_opts = sorted(df_clean["thoi_tiet"].unique())
            selected_weather = st.multiselect("Điều kiện Thời tiết:", weather_opts, default=weather_opts)
        with col_f3:
            vehicle_opts = sorted(df_clean["loai_xe"].unique())
            selected_vehicles = st.multiselect("Loại Phương tiện:", vehicle_opts, default=vehicle_opts)

    filter_mask = (
        df_clean["gio"].between(hour_range[0], hour_range[1]) &
        df_clean["thoi_tiet"].isin(selected_weather) &
        df_clean["loai_xe"].isin(selected_vehicles)
    )
    df_filtered = df_clean[filter_mask].copy()

    if len(df_filtered) == 0:
        st.warning("Không có dữ liệu thỏa mãn bộ lọc hiện tại. Vui lòng nới rộng tiêu chí lọc!")
        return

    n_curr = len(df_filtered)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    avg_speed = df_filtered["toc_do_ghi_nhan"].mean()
    pct_over_speed = (df_filtered["toc_do_vuot_muc"] > 0).mean() * 100
    peak_count = df_filtered["la_gio_cao_diem"].sum()

    kpi1.metric("Số vi phạm đã lọc", f"{n_curr:,}")
    kpi2.metric("Tốc độ trung bình", f"{avg_speed:.1f} km/h")
    kpi3.metric("Tỷ lệ vượt tốc độ", f"{pct_over_speed:.1f}%")
    kpi4.metric("Vi phạm giờ cao điểm", f"{peak_count:,} ({peak_count/n_curr*100:.1f}%)")

    st.markdown("---")

    # ------------------ MODULE 3: DUAL HISTOGRAM WITH LOG SCALE ------------------
    st.subheader("1. Phân phối Tốc độ Vi phạm: Tuyến tính vs Trục Logarit")
    
    col_hist_ctrl, col_hist_plot = st.columns([1, 3])
    with col_hist_ctrl:
        st.markdown("**Tùy chỉnh Thang đo & Bins:**")
        n_bins = st.slider("Số lượng bins:", min_value=15, max_value=60, value=30, step=5)
        color_scheme = st.selectbox("Màu chủ đạo biểu đồ kép:", ["#2563EB (Royal Blue)", "#7C3AED (Purple)", "#059669 (Emerald)"])
        chosen_color = color_scheme.split()[0]

    with col_hist_plot:
        fig_dual_hist = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Thang đo Tuyến tính (Linear Scale, n={n_curr})",
                f"Thang đo Logarit (Log10 Scale, n={n_curr})"
            ],
            horizontal_spacing=0.10
        )
        
        # Subplot 1: Linear Scale
        fig_dual_hist.add_trace(
            go.Histogram(
                x=df_filtered["toc_do_ghi_nhan"],
                nbinsx=n_bins,
                marker_color=chosen_color,
                opacity=0.75,
                name="Tuyến tính (km/h)",
                hovertemplate="Tốc độ: %{x} km/h<br>Số lượng: %{y}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Subplot 2: Log10 Scale
        log_speeds = np.log10(df_filtered["toc_do_ghi_nhan"].clip(lower=1.0))
        fig_dual_hist.add_trace(
            go.Histogram(
                x=log_speeds,
                nbinsx=n_bins,
                marker_color="#DC2626",
                opacity=0.75,
                name="Log10 Tốc độ",
                hovertemplate="Log10(Tốc độ): %{x:.2f}<br>Số lượng: %{y}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Add vertical line for mean on both
        mean_spd = df_filtered["toc_do_ghi_nhan"].mean()
        fig_dual_hist.add_vline(x=mean_spd, line_dash="dash", line_color="black", row=1, col=1,
                                annotation_text=f"Mean: {mean_spd:.1f} km/h", annotation_position="top left")
        fig_dual_hist.add_vline(x=np.log10(mean_spd), line_dash="dash", line_color="black", row=1, col=2,
                                annotation_text=f"Log10(Mean): {np.log10(mean_spd):.2f}", annotation_position="top left")
        
        fig_dual_hist.update_xaxes(title_text="Tốc độ ghi nhận (km/h)", row=1, col=1)
        fig_dual_hist.update_xaxes(title_text="Log₁₀ (Tốc độ ghi nhận)", row=1, col=2)
        fig_dual_hist.update_yaxes(title_text="Số lượng vi phạm", row=1, col=1)
        fig_dual_hist.update_yaxes(title_text="Số lượng vi phạm", row=1, col=2)
        
        fig_dual_hist.update_layout(
            title_text=f"Phân tích Kép Phân phối Tốc độ Vi phạm: Tuyến tính vs Log10 (n={n_curr})",
            height=420,
            margin=dict(l=20, r=20, t=50, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_dual_hist, use_container_width=True)
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_curr:,}")

    # ------------------ MODULE 2: BAR CHART WITH ERROR BARS (BOOTSTRAP CI 95%) ------------------
    st.subheader("2. So sánh Tốc độ Vượt ngưỡng Trung bình có Vạch Sai số (Error Bars)")
    
    # Compute Bootstrap CI 95% across Vehicle types
    vehicle_types_present = sorted(df_filtered["loai_xe"].unique())
    means_list, err_minus_list, err_plus_list, ns_list = [], [], [], []
    
    for v in vehicle_types_present:
        v_data = df_filtered[df_filtered["loai_xe"] == v]["toc_do_vuot_muc"].values
        m_val, ci_low, ci_high = compute_bootstrap_ci(v_data, n_bootstraps=1000, ci=0.95)
        means_list.append(m_val)
        err_minus_list.append(m_val - ci_low)
        err_plus_list.append(ci_high - m_val)
        ns_list.append(len(v_data))
        
    labels_with_n = [f"{v}<br>(n={n:,})" for v, n in zip(vehicle_types_present, ns_list)]
    
    fig_error_bar = go.Figure(
        data=go.Bar(
            x=labels_with_n,
            y=means_list,
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus_list,
                arrayminus=err_minus_list,
                visible=True,
                color="#0F172A",
                thickness=2,
                width=8
            ),
            marker=dict(
                color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"][:len(vehicle_types_present)],
                opacity=0.85,
                line=dict(color="#1E293B", width=1.5)
            ),
            text=[f"{m:.2f} km/h" for m in means_list],
            textposition="outside",
            hovertemplate="Phương tiện: %{x}<br>Tốc độ vượt TB: %{y:.2f} km/h<extra></extra>"
        )
    )
    
    fig_error_bar.update_layout(
        title=f"Tốc độ Vượt ngưỡng Trung bình theo Phương tiện kèm Vạch Sai số Bootstrap 95% (n={n_curr})",
        xaxis_title="Loại Phương tiện",
        yaxis_title="Tốc độ Vượt ngưỡng Trung bình: Δv = max(0, v - v_limit) [km/h]",
        height=400,
        margin=dict(l=20, r=20, t=50, b=30)
    )
    st.plotly_chart(fig_error_bar, use_container_width=True)
    st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_curr:,}")

    # ------------------ MODULE 4: SPEARMAN CORRELATION HEATMAP ------------------
    st.subheader("3. Ma trận Tương quan Phi tham số Spearman")
    
    corr_cols = ["gio", "toc_do_ghi_nhan", "toc_do_gioi_han", "toc_do_vuot_muc"]
    corr_labels = ["Giờ trong ngày", "Tốc độ ghi nhận", "Tốc độ giới hạn", "Mức độ vượt tốc độ"]
    
    spearman_matrix = df_filtered[corr_cols].corr(method="spearman").values
    
    fig_spearman = go.Figure(
        data=go.Heatmap(
            z=spearman_matrix,
            x=corr_labels,
            y=corr_labels,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            text=np.round(spearman_matrix, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 13, "color": "black"},
            colorbar=dict(title="Hệ số Spearman (ρ)", tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0]),
            hoverongaps=False
        )
    )
    
    fig_spearman.update_layout(
        title=f"Ma trận Tương quan Phi tham số Spearman giữa các Biến Định lượng (n={n_curr})",
        height=420,
        margin=dict(l=20, r=20, t=50, b=30)
    )
    st.plotly_chart(fig_spearman, use_container_width=True)
    st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_curr:,}")

    st.markdown("---")

    # ------------------ MODULE 5: ALGORITHM EVALUATION (SILHOUETTE SCORE SWEEP) ------------------
    st.subheader("4. Bản đồ Điểm đen & Trực quan hóa Độ gãy Thuật toán DBSCAN")
    st.markdown("""
    Để lựa chọn siêu tham số bán kính $\\epsilon$ một cách khoa học thay vì ngẫu nhiên, hệ thống tính toán 
    chỉ số **Silhouette Score** qua các dải bán kính để xác định điểm cân bằng tối ưu giữa độ kết tụ cụm và tỷ lệ nhiễu.
    """)

    col_db_ctrl1, col_db_ctrl2, col_db_ctrl3 = st.columns(3)
    with col_db_ctrl1:
        eps_meters = st.slider("Bán kính lân cận ε (Epsilon, mét):", min_value=150.0, max_value=1200.0, value=450.0, step=50.0)
    with col_db_ctrl2:
        min_pts = st.slider("Số điểm tối thiểu MinPts (Core Points):", min_value=5, max_value=50, value=15, step=1)
    with col_db_ctrl3:
        max_render_points = st.selectbox("Giới hạn hiển thị trên bản đồ (MarkerCluster):", [1000, 2000, 3000, 5000], index=1)

    df_for_clustering = df_filtered.copy()
    if len(df_for_clustering) > max_render_points:
        df_clustering_sample = df_for_clustering.sample(n=max_render_points, random_state=42).copy()
    else:
        df_clustering_sample = df_for_clustering.copy()

    n_sample_cluster = len(df_clustering_sample)
    coords = df_clustering_sample[["vi_do", "kinh_do"]].values

    # Run Silhouette Curve Evaluation
    with st.spinner("Đang tính toán đường cong Silhouette Score cho hội đồng khoa học..."):
        df_sil = compute_dbscan_silhouette_curve(coords, min_samples=min_pts)

    # Plot Algorithm Evaluation Line Chart
    fig_sil = go.Figure()
    fig_sil.add_trace(
        go.Scatter(
            x=df_sil["eps_meters"],
            y=df_sil["silhouette"],
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=9, color="#1D4ED8", symbol="diamond"),
            hovertemplate="Bán kính ε: %{x}m<br>Silhouette Score: %{y:.3f}<extra></extra>"
        )
    )
    
    # Highlight current selected epsilon
    fig_sil.add_vline(
        x=eps_meters,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Cấu hình Chọn: ε = {eps_meters}m",
        annotation_position="top right"
    )
    
    fig_sil.update_layout(
        title=f"Độ gãy Thuật toán: Điểm Silhouette Score theo Bán kính Epsilon (n={n_sample_cluster})",
        xaxis_title="Bán kính Lân cận ε (Epsilon, mét)",
        yaxis_title="Silhouette Score (Khoảng cách Haversine)",
        height=350,
        margin=dict(l=20, r=20, t=50, b=30)
    )
    st.plotly_chart(fig_sil, use_container_width=True)
    st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_sample_cluster:,}")

    # Run DBSCAN with selected configuration
    cluster_labels = execute_dbscan_clustering(coords, eps_meters=eps_meters, min_samples=min_pts)
    df_clustering_sample["cluster"] = cluster_labels

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())

    st.info(f"📍 **Kết quả Phân cụm DBSCAN:** Phát hiện **{n_clusters}** cụm điểm đen mật độ cao | **{n_noise}** điểm sự cố ngẫu nhiên / nhiễu ({n_noise/len(df_clustering_sample)*100:.1f}%).")

    # Interactive Folium Map with MarkerCluster
    if HAS_FOLIUM:
        center_lat = float(df_clustering_sample["vi_do"].median())
        center_lon = float(df_clustering_sample["kinh_do"].median())
        
        # Sử dụng OpenStreetMap làm nền mặc định (100% Free, Không yêu cầu API Key, không watermark)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="OpenStreetMap"
        )
        
        # Thêm các lớp bản đồ phụ trợ miễn phí (không yêu cầu API Key)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
            name="Bản đồ Vệ tinh (Esri Satellite)",
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a>',
            name="Bản đồ Giao thông & Địa hình (OSM HOT)",
            control=True
        ).add_to(m)
        
        palette = [
            "#E11D48", "#2563EB", "#059669", "#D97706", "#7C3AED",
            "#DB2777", "#0891B2", "#4F46E5", "#CA8A04", "#0D9488"
        ]
        
        marker_cluster = MarkerCluster(
            name="Cụm điểm đen vi phạm",
            overlay=True,
            control=True
        ).add_to(m)
        
        for _, row in df_clustering_sample.iterrows():
            c_id = row["cluster"]
            if c_id == -1:
                color = "#94A3B8"
                cluster_desc = "Nhiễu ngẫu nhiên (Noise)"
                rad = 3
            else:
                color = palette[c_id % len(palette)]
                cluster_desc = f"Điểm đen Cụm #{c_id}"
                rad = 6
                
            popup_html = f"""
            <div style="font-family:sans-serif; font-size:12px; width:220px;">
                <b style="color:{color}; font-size:13px;">{cluster_desc}</b><br>
                <b>Mã VP:</b> {row['ma_vi_pham']}<br>
                <b>Thời gian:</b> {row['thoi_gian']}<br>
                <b>Phương tiện:</b> {row['loai_xe']}<br>
                <b>Hành vi:</b> {row['loai_vi_pham']}<br>
                <b>Tốc độ:</b> {row['toc_do_ghi_nhan']} km/h (Giới hạn: {row['toc_do_gioi_han']})<br>
                <b>Thời tiết:</b> {row['thoi_tiet']}
            </div>
            """
            
            folium.CircleMarker(
                location=[row["vi_do"], row["kinh_do"]],
                radius=rad,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{row['loai_vi_pham']} ({row['toc_do_ghi_nhan']} km/h) - {cluster_desc}"
            ).add_to(marker_cluster)
            
        for c in range(n_clusters):
            c_df = df_clustering_sample[df_clustering_sample["cluster"] == c]
            if len(c_df) >= min_pts:
                c_lat = float(c_df["vi_do"].mean())
                c_lon = float(c_df["kinh_do"].mean())
                folium.Circle(
                    location=[c_lat, c_lon],
                    radius=eps_meters,
                    color=palette[c % len(palette)],
                    weight=1.5,
                    fill=True,
                    fill_opacity=0.1,
                    tooltip=f"Tâm Điểm Đen Cụm #{c} (N={len(c_df)})"
                ).add_to(m)
        
        folium.LayerControl(position="topright", collapsed=True).add_to(m)
        st_folium(m, width="100%", height=550, returned_objects=[])
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích không gian trên tập n = {n_sample_cluster:,} | Lớp nền: OpenStreetMap & Esri")
    else:
        st.info("ℹ️ Chế độ bản đồ dự phòng Plotly (Để trải nghiệm bản đồ Folium + MarkerCluster, vui lòng cài đặt: pip install folium streamlit-folium).")
        fig_map = px.scatter_mapbox(
            df_clustering_sample,
            lat="vi_do",
            lon="kinh_do",
            color=df_clustering_sample["cluster"].astype(str),
            hover_name="loai_vi_pham",
            hover_data=["toc_do_ghi_nhan", "loai_xe", "thoi_tiet", "dia_diem"],
            zoom=11,
            title=f"Bản đồ Phân cụm Điểm đen Giao thông (Plotly Mapbox, n={n_sample_cluster})"
        )
        fig_map.update_layout(mapbox_style="open-street-map", height=550)
        st.plotly_chart(fig_map, use_container_width=True)

    if n_clusters > 0:
        st.markdown("**Bảng Thống kê Chi tiết các Cụm Điểm đen phát hiện:**")
        summary_rows = []
        for c in range(n_clusters):
            c_data = df_clustering_sample[df_clustering_sample["cluster"] == c]
            top_violation = c_data["loai_vi_pham"].mode().iloc[0] if len(c_data) > 0 else "N/A"
            top_vehicle = c_data["loai_xe"].mode().iloc[0] if len(c_data) > 0 else "N/A"
            summary_rows.append({
                "Cluster ID": f"Cụm #{c}",
                "Số Vụ Vi phạm": len(c_data),
                "Tỷ lệ trên mẫu (%)": round(len(c_data) / len(df_clustering_sample) * 100, 1),
                "Tốc độ TB (km/h)": round(c_data["toc_do_ghi_nhan"].mean(), 1),
                "Tốc độ Vượt TB (km/h)": round(c_data["toc_do_vuot_muc"].mean(), 1),
                "Hành vi Chủ đạo": top_violation,
                "Phương tiện Phổ biến": top_vehicle,
                "Tâm Vĩ độ": round(c_data["vi_do"].mean(), 5),
                "Tâm Kinh độ": round(c_data["kinh_do"].mean(), 5),
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# 7. VIEW PAGE 4: STATISTICAL TESTING & SENSITIVITY ANALYSIS (MODULE 6)
# ==============================================================================
def render_page_statistical_testing(df_clean: pd.DataFrame, df_raw: pd.DataFrame):
    st.title("🧪 Kiểm định Giả thuyết Thống kê & Phân tích Độ nhạy")
    st.caption("Kiểm định định lượng tính độc lập, sai khác giữa các nhóm và phân tích độ nhạy của tham số vị trí")

    n_clean = len(df_clean)
    n_raw = len(df_raw)

    test_type = st.radio(
        "Chọn Phân hệ Thống kê Suy diễn:",
        [
            "Kiểm định Tính Độc lập Chi-Square (χ² Independence Test)",
            "Kiểm định Trung bình Hai Mẫu Độc lập (Welch's Two-Sample T-Test)",
            "Phân tích Độ nhạy Thống kê (Sensitivity Analysis)"
        ],
        horizontal=True
    )

    st.markdown("---")

    # ------------------ OPTION 1: CHI-SQUARE TEST ------------------
    if "Chi-Square" in test_type:
        st.subheader("1. Kiểm định Tính Độc lập Chi-Square (χ² Test of Independence)")
        st.markdown("""
        Kiểm định xem có **mối quan hệ phụ thuộc có ý nghĩa thống kê** giữa hai biến phân loại hay không:
        - Giả thuyết $H_0$: Hai biến độc lập với nhau.
        - Giả thuyết $H_1$: Hai biến có mối tương quan phụ thuộc.
        """)

        col_chi_v1, col_chi_v2 = st.columns(2)
        categorical_cols = ["thoi_tiet", "loai_xe", "khung_gio_nhom", "la_cuoi_tuan"]
        
        with col_chi_v1:
            var_x = st.selectbox("Chọn Biến Độc lập (X):", categorical_cols, index=0)
        with col_chi_v2:
            target_cols = ["loai_vi_pham", "la_gio_cao_diem"]
            var_y = st.selectbox("Chọn Biến Phụ thuộc (Y):", target_cols, index=0)

        crosstab = pd.crosstab(df_clean[var_x], df_clean[var_y])
        crosstab_with_totals = pd.crosstab(df_clean[var_x], df_clean[var_y], margins=True, margins_name="Tổng")

        st.markdown(f"**Bảng Tần số Kết hợp Thực tế (Contingency Table, n={n_clean}):**")
        st.dataframe(crosstab_with_totals, use_container_width=True)

        chi2_stat, p_val, dof, expected = stats.chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        r, k = crosstab.shape
        cramers_v = np.sqrt(chi2_stat / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chi-Square (χ²)", f"{chi2_stat:.3f}")
        m2.metric("Bậc tự do (df)", f"{dof}")
        m3.metric("P-value", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}")
        m4.metric("Kích thước tác động (Cramér's V)", f"{cramers_v:.3f}")

        alpha = 0.05
        if p_val < alpha:
            st.success(
                f"✅ **KẾT LUẬN: BÁC BỎ GIẢ THUYẾT H₀ (p-value = {p_val:.4e} < 0.05)**.\n\n"
                f"Tồn tại mối quan hệ phụ thuộc có ý nghĩa thống kê ở mức ý nghĩa 5% giữa **{var_x}** và **{var_y}**. "
                f"Hành vi vi phạm giao thông phân hóa rõ rệt theo các điều kiện của biến độc lập (Cramér's V = {cramers_v:.3f})."
            )
        else:
            st.error(
                f"❌ **KẾT LUẬN: CHƯA ĐỦ BẰNG CHỨNG ĐỂ BÁC BỎ H₀ (p-value = {p_val:.4f} >= 0.05)**.\n\n"
                f"Không có sự phụ thuộc có ý nghĩa thống kê giữa **{var_x}** và **{var_y}**. "
                f"Sự khác biệt quan sát được có thể do ngẫu nhiên mẫu."
            )

        crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
        fig_bar = px.bar(
            crosstab_pct.reset_index().melt(id_vars=var_x, value_name="Phần trăm (%)", var_name=var_y),
            x=var_x,
            y="Phần trăm (%)",
            color=var_y,
            title=f"Tỷ lệ Phân phối Điều kiện: {var_y} theo từng nhóm {var_x} (n={n_clean})",
            barmode="stack",
            text_auto=".1f"
        )
        fig_bar.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_clean:,}")

    # ------------------ OPTION 2: TWO-SAMPLE WELCH'S T-TEST ------------------
    elif "Welch's Two-Sample T-Test" in test_type:
        st.subheader("2. Kiểm định Trung bình Hai Mẫu Độc lập (Welch's Two-Sample T-Test)")
        st.markdown("""
        So sánh giá trị trung bình của biến liên tục giữa 2 nhóm độc lập (không giả định phương sai bằng nhau):
        - Giả thuyết $H_0: \\mu_1 = \\mu_2$ (Không có sai khác trung bình giữa 2 nhóm).
        - Giả thuyết $H_1: \\mu_1 \\ne \\mu_2$ (Tồn tại sự khác biệt có ý nghĩa thống kê).
        """)

        col_t_num, col_t_grp = st.columns(2)
        with col_t_num:
            num_var = st.selectbox("Chọn Biến Định lượng Cần Kiểm Định:", ["toc_do_vuot_muc", "toc_do_ghi_nhan"])
        with col_t_grp:
            grouping_scheme = st.selectbox(
                "Chọn Cặp Nhóm Đối Sánh:",
                [
                    "Xe máy vs Ô tô con (Phân loại phương tiện)",
                    "Ngày thường vs Cuối tuần (Thời gian)",
                    "Giờ cao điểm vs Giờ bình thường"
                ]
            )

        if "Xe máy vs Ô tô con" in grouping_scheme:
            g1_name, g2_name = "Xe máy", "Ô tô con"
            g1_data = df_clean[df_clean["loai_xe"] == g1_name][num_var].dropna()
            g2_data = df_clean[df_clean["loai_xe"] == g2_name][num_var].dropna()
        elif "Ngày thường vs Cuối tuần" in grouping_scheme:
            g1_name, g2_name = "Ngày thường", "Cuối tuần"
            g1_data = df_clean[~df_clean["la_cuoi_tuan"]][num_var].dropna()
            g2_data = df_clean[df_clean["la_cuoi_tuan"]][num_var].dropna()
        else:
            g1_name, g2_name = "Giờ cao điểm", "Giờ bình thường"
            g1_data = df_clean[df_clean["la_gio_cao_diem"]][num_var].dropna()
            g2_data = df_clean[~df_clean["la_gio_cao_diem"]][num_var].dropna()

        n1, n2 = len(g1_data), len(g2_data)
        m1, m2 = g1_data.mean(), g2_data.mean()
        s1, s2 = g1_data.std(ddof=1), g2_data.std(ddof=1)

        summary_t_df = pd.DataFrame([
            {"Nhóm": g1_name, "Kích thước mẫu (N)": n1, "Trung bình (Mean)": round(m1, 2), "Độ lệch chuẩn (Std)": round(s1, 2)},
            {"Nhóm": g2_name, "Kích thước mẫu (N)": n2, "Trung bình (Mean)": round(m2, 2), "Độ lệch chuẩn (Std)": round(s2, 2)}
        ])
        st.table(summary_t_df)

        t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=False)
        dof_welch = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1 - 1) + (s2**2/n2)**2/(n2 - 1))
        diff_mean = m1 - m2

        tm1, tm2, tm3, tm4 = st.columns(4)
        tm1.metric("T-Statistic", f"{t_stat:.3f}")
        tm2.metric("Chênh lệch TB (Δμ)", f"{diff_mean:.2f} km/h")
        tm3.metric("Bậc tự do (Welch df)", f"{dof_welch:.1f}")
        tm4.metric("P-value", f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}")

        if p_val < 0.05:
            st.success(
                f"✅ **KẾT LUẬN: BÁC BỎ GIẢ THUYẾT H₀ (p-value = {p_val:.4e} < 0.05)**.\n\n"
                f"Tồn tại sự khác biệt có ý nghĩa thống kê về **{num_var}** giữa nhóm **{g1_name}** "
                f"(Mean = {m1:.2f}) và nhóm **{g2_name}** (Mean = {m2:.2f})."
            )
        else:
            st.error(
                f"❌ **KẾT LUẬN: CHƯA ĐỦ BẰNG CHỨNG ĐỂ BÁC BỎ H₀ (p-value = {p_val:.4f} >= 0.05)**.\n\n"
                f"Không có sự khác biệt có ý nghĩa thống kê giữa hai nhóm ở mức ý nghĩa 5%."
            )

        plot_df = pd.DataFrame({
            "Giá trị": np.concatenate([g1_data.values, g2_data.values]),
            "Nhóm": [g1_name]*n1 + [g2_name]*n2
        })
        fig_violin = px.violin(
            plot_df,
            x="Nhóm",
            y="Giá trị",
            color="Nhóm",
            box=True,
            points="outliers",
            title=f"Biểu đồ Violin & Boxplot So sánh Phân phối: {num_var} (n={n1 + n2})",
            labels={"Giá trị": f"{num_var} (km/h)"}
        )
        fig_violin.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=30), showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n1 + n2:,}")

    # ------------------ MODULE 6: SENSITIVITY ANALYSIS ------------------
    else:
        st.subheader("3. Phân tích Độ nhạy Thống kê (Sensitivity Analysis)")
        st.markdown("""
        Đánh giá độ bền vững (robustness) của các chỉ tiêu thống kê trung tâm khi thay đổi giả định làm sạch dữ liệu.
        Dưới đây là so sánh chỉ tiêu **Trung vị Tốc độ Vi phạm (Median Speed)** dưới 3 kịch bản bộ lọc khác nhau:
        - **Kịch bản (a)**: Toàn bộ dữ liệu thô (chứa cả ngoại lai GPS và xung cảm biến).
        - **Kịch bản (b)**: Chỉ loại bỏ dữ liệu GPS lỗi tọa độ ngoài khu vực đô thị.
        - **Kịch bản (c)**: Loại bỏ lỗi GPS + Cắt bỏ 1% Outlier tốc độ cao nhất (ngưỡng Phân vị 99th).
        """)

        # Scenario Computations
        # A. Raw
        med_a = float(df_raw["toc_do_ghi_nhan"].median())
        mean_a = float(df_raw["toc_do_ghi_nhan"].mean())
        n_a = len(df_raw)

        # B. GPS Clean Only
        valid_gps = (
            df_raw["vi_do"].between(GPS_BOUNDS["lat_min"], GPS_BOUNDS["lat_max"]) &
            df_raw["kinh_do"].between(GPS_BOUNDS["lon_min"], GPS_BOUNDS["lon_max"])
        )
        df_b = df_raw[valid_gps]
        med_b = float(df_b["toc_do_ghi_nhan"].median())
        mean_b = float(df_b["toc_do_ghi_nhan"].mean())
        n_b = len(df_b)

        # C. GPS Clean + 99th Percentile Outlier Trimming
        p99_speed = df_b["toc_do_ghi_nhan"].quantile(0.99)
        df_c = df_b[df_b["toc_do_ghi_nhan"] <= p99_speed]
        med_c = float(df_c["toc_do_ghi_nhan"].median())
        mean_c = float(df_c["toc_do_ghi_nhan"].mean())
        n_c = len(df_c)

        # Build Comparison DataFrame
        scenarios = [
            f"(a) Dữ liệu Thô Toàn bộ<br>(n={n_a:,})",
            f"(b) Chỉ Bỏ Lỗi GPS<br>(n={n_b:,})",
            f"(c) Bỏ GPS + Cắt Outlier P99<br>(n={n_c:,})"
        ]
        medians = [med_a, med_b, med_c]
        means = [mean_a, mean_b, mean_c]
        
        # Horizontal Bar Chart for Sensitivity Analysis
        fig_sens = go.Figure()
        
        fig_sens.add_trace(
            go.Bar(
                y=scenarios,
                x=medians,
                orientation="h",
                name="Trung vị (Median)",
                marker=dict(color="#2563EB", opacity=0.85),
                text=[f"Median: {m:.2f} km/h" for m in medians],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=12)
            )
        )
        
        fig_sens.add_trace(
            go.Scatter(
                y=scenarios,
                x=means,
                mode="markers+text",
                name="Trung bình (Mean - Bị kéo bởi ngoại lai)",
                marker=dict(color="#DC2626", size=14, symbol="diamond"),
                text=[f"Mean: {m:.2f}" for m in means],
                textposition="top center",
                textfont=dict(color="#DC2626", size=11, family="Arial Black")
            )
        )
        
        fig_sens.update_layout(
            title=f"Phân tích Độ nhạy: Trung vị vs Trung bình qua 3 Kịch bản Lọc Dữ liệu (n={n_raw})",
            xaxis_title="Tốc độ ghi nhận (km/h)",
            yaxis_title="Kịch bản Lọc Dữ liệu",
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", y=1.15, x=0.1)
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.caption(f"Nguồn dữ liệu: Dữ liệu mô phỏng | Phân tích có điều kiện trên tập n = {n_raw:,}")

        st.markdown(f"""
        <div class="academic-box">
            <b>💡 Luận điểm Khoa học (Academic Insight):</b><br>
            - <strong>Tính Bền vững của Trung vị (Median Robustness):</strong> Giá trị Trung vị dao động không đáng kể giữa 3 kịch bản 
            (từ <code>{med_a:.1f} km/h</code> đến <code>{med_c:.1f} km/h</code>, biến thiên &lt; 0.5%), chứng minh tính bền vững cao trước các giá trị ngoại lai cảm biến.<br>
            - <strong>Độ Nhạy của Trung bình (Mean Sensitivity):</strong> Ngược lại, giá trị Trung bình (Mean) bị phóng đại mạnh ở kịch bản thô 
            (<code>{mean_a:.2f} km/h</code>) do ảnh hưởng của các xung tốc độ siêu thanh vô lý (285.5 km/h), và chỉ hội tụ thực tế (<code>{mean_c:.2f} km/h</code>) 
            sau khi cắt bỏ 1% đuôi ngoại lai cao nhất.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 8. MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=70)
        st.title("TRAFFIC AI SYSTEM")
        st.caption("Intelligent Transport Analytics v2.0 (Academic Edition)")
        
        st.markdown("---")
        
        page = st.radio(
            "ĐIỀU HƯỚNG TRANG:",
            [
                "1. Cơ sở lý thuyết & DGP",
                "2. Pipeline & Tiền xử lý",
                "3. Khai phá & Bản đồ Không gian",
                "4. Kiểm định Thống kê"
            ],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("**Trạng thái Ứng dụng:**")
        st.success("🟢 Production Ready (Academic)")
        st.markdown("""
        - **DGP Size:** 5,000 records
        - **Flowchart:** Graphviz Dynamic Flow
        - **Subplots:** Plotly Log10 & Linear
        - **Error Bars:** 95% Bootstrap CI
        - **Sensitivity:** Median Robustness
        """)
        
        st.markdown("---")
        st.caption("Developed by Senior Data Scientist © 2024")

    # Ingestion & Pre-cleaning
    df_raw = load_raw_dataset()
    df_clean, _ = clean_traffic_pipeline(df_raw, apply_filters=True)

    # Routing
    if "1. Cơ sở lý thuyết" in page:
        render_page_overview_theory()
    elif "2. Pipeline & Tiền xử lý" in page:
        render_page_data_pipeline(df_raw)
    elif "3. Khai phá & Bản đồ" in page:
        render_page_spatial_eda(df_clean)
    elif "4. Kiểm định Thống kê" in page:
        render_page_statistical_testing(df_clean, df_raw)


if __name__ == "__main__":
    main()
