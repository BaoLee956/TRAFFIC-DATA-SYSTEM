# 🚦 TRAFFIC DATA SYSTEM: Urban Traffic Violation Analytics & Decision Support

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-DBSCAN-F7931E.svg)](https://scikit-learn.org/)
[![Folium](https://img.shields.io/badge/Folium-Geospatial-77B800.svg)](https://python-visualization.github.io/folium/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Viz-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Hệ thống phân tích và khai phá dữ liệu vi phạm giao thông đô thị thông minh (Intelligent Transport Systems - ITS)**, tích hợp quy trình Khoa học Dữ liệu chuẩn mực: Mô hình sinh dữ liệu (DGP), Tiền xử lý & làm sạch viễn thám GPS, Phân cụm mật độ không gian (DBSCAN với metric Haversine) và Kiểm định giả thuyết thống kê suy diễn ($\chi^2$-test, Welch's t-test).

---

## 📌 Mục lục

1. [Tổng quan Dự án](#-tổng-quan-dự-án)
2. [Cơ sở Lý thuyết & Mô hình Sinh Dữ liệu (DGP)](#-cơ-sở-lý-thuyết--mô-hình-sinh-dữ-liệu-dgp)
3. [Kiến trúc Ứng dụng (Multi-Page Navigation)](#-kiến-trúc-ứng-dụng-multi-page-navigation)
4. [Công nghệ Sử dụng (Tech Stack)](#-công-nghệ-sử-dụng-tech-stack)
5. [Cấu trúc Thư mục](#-cấu-trúc-thư-mục)
6. [Hướng dẫn Cài đặt & Khởi chạy](#-hướng-dẫn-cài-đặt--khởi-chạy)
7. [Tối ưu hóa Kỹ thuật & Hiệu năng](#-tối-ưu-hóa-kỹ-thuật--hiệu-năng)
8. [Tác giả & Đóng góp](#-tác-giả--đóng-góp)

---

## 🎯 Tổng quan Dự án

Trong quản lý giao thông đô thị hiện đại, việc chuyển đổi từ dữ liệu camera/cảm biến thô sang các quyết định điều phối thực tế đòi hỏi một hệ thống có tính giải thích toán học và độ tin cậy thống kê cao. 

**TRAFFIC DATA SYSTEM** không dừng lại ở việc biểu diễn đồ thị thông thường mà xây dựng một luồng suy luận khắt khe:
- **Tự động phát hiện và làm sạch lỗi cảm biến**: Lọc nhiễu GPS drift, tốc độ âm hoặc phi vật lý.
- **Phân cụm điểm đen mật độ cao (Blackspots)**: Ứng dụng DBSCAN với khoảng cách mặt cầu Haversine để nhận diện các nút giao có rủi ro tai nạn/vi phạm cao.
- **Kiểm định suy diễn định lượng**: Xác nhận có ý nghĩa thống kê hay không đối với các tác động của thời tiết, loại phương tiện và hành vi vi phạm.

---

## 📐 Cơ sở Lý thuyết & Mô hình Sinh Dữ liệu (DGP)

Hệ thống tích hợp mô hình **Data Generating Process (DGP)** tự động khi chưa có dữ liệu camera thực tế:

### 1. Quá trình Thời gian (Inhomogeneous Poisson Process)
Số lượng sự kiện vi phạm đến theo thời gian tuân theo phân phối Poisson không thuần nhất với hàm cường độ $\lambda(t)$ đạt cực đại tại các khung giờ cao điểm sáng ($07\text{h}30 - 08\text{h}30$), chiều ($17\text{h}30 - 18\text{h}30$) và ban đêm:
$$P(N(t + \Delta t) - N(t) = k) = \frac{\left(\int_t^{t+\Delta t} \lambda(u) du\right)^k e^{-\int_t^{t+\Delta t} \lambda(u) du}}{k!}$$

### 2. Khoảng cách Trắc địa Haversine trên Ellipsoid Trái Đất
Tính toán khoảng cách thực tế giữa hai điểm GPS $(\phi_1, \lambda_1)$ và $(\phi_2, \lambda_2)$ với $R \approx 6,371.0088\text{ km}$:
$$d_{\text{haversine}} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1 \cos\phi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

### 3. Thuật toán Phân cụm Mật độ DBSCAN
Nhận diện cụm mật độ tự nhiên không cần khai báo trước số lượng $k$:
$$N_\epsilon(p) = \{q \in \mathcal{D} \mid d_{\text{haversine}}(p, q) \le \epsilon\}, \quad |N_\epsilon(p)| \ge \text{MinPts}$$

### 4. Kiểm định Thống kê Suy diễn
- **Chi-Square Independence Test**:
  $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{R_i \cdot C_j}{N}$$
- **Welch's Two-Sample T-Test** (không giả định bằng nhau về phương sai giữa các nhóm phương tiện):
  $$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

---

## 📱 Kiến trúc Ứng dụng (Multi-Page Navigation)

Ứng dụng chia thành 4 phân hệ chính điều hướng qua Sidebar:

| Phân hệ | Chức năng chính |
| :--- | :--- |
| **Page 1: Cơ sở lý thuyết & DGP** | Trình bày bối cảnh ITS, công thức toán học (`st.latex`), 3 giả thuyết nghiên cứu ($H_1, H_2, H_3$) và Data Dictionary chuẩn xác. |
| **Page 2: Pipeline & Tiền xử lý** | Hai tabs **Raw Data** & **Cleaned Data**. Toggle bật/tắt bộ lọc nhiễu GPS Bounding Box, dị biệt tốc độ và hiển thị `st.code()` pipeline minh bạch. |
| **Page 3: Khai phá & Bản đồ** | Global Filters (Giờ, Thời tiết, Phương tiện). Biểu đồ phân phối tốc độ Plotly Histogram tùy biến bin size. Bản đồ điểm đen giao thông Folium tích hợp **DBSCAN** và **MarkerCluster**. |
| **Page 4: Kiểm định Thống kê** | Lựa chọn biến tương tác, tự động chạy $\chi^2$-test hoặc Welch's t-test, in Contingency table, hiển thị $p$-value bằng `st.metric()`, kết luận màu xanh (`st.success`) khi $p < 0.05$ và màu đỏ (`st.error`) khi $p \ge 0.05$. |

---

## 🛠️ Công nghệ Sử dụng (Tech Stack)

- **Ngôn ngữ**: Python 3.10+
- **Giao diện & Ứng dụng**: [Streamlit](https://streamlit.io/)
- **Thao tác Dữ liệu**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Hệ thống Không gian & Bản đồ**: [Folium](https://python-visualization.github.io/folium/), [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium)
- **Trực quan hóa Dữ liệu**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Học máy & Phân cụm**: [Scikit-learn](https://scikit-learn.org/) (DBSCAN with BallTree & Haversine metric)
- **Thống kê Suy diễn**: [SciPy](https://scipy.org/) (`scipy.stats.chi2_contingency`, `scipy.stats.ttest_ind`)

---

## 📂 Cấu trúc Thư mục

```text
TRAFFIC-DATA-SYSTEM/
├── .gitignore                     # Quy tắc loại trừ tệp rác, cache, môi trường ảo
├── requirements.txt               # Danh sách dependencies chuẩn phiên bản
├── app.py                         # Toàn bộ mã nguồn ứng dụng Streamlit (Production-ready)
├── traffic_violations.csv         # Dữ liệu gốc nạp vào ứng dụng (5,000 bản ghi thô)
├── traffic_violations_raw.csv     # Bản sao lưu dữ liệu thô phục vụ kiểm toán
├── traffic_violations_cleaned.csv # Dữ liệu sau tiền xử lý & làm sạch (4,815 bản ghi)
└── README.md                      # Tài liệu chi tiết về dự án và hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Clone kho mã nguồn về máy
```bash
git clone https://github.com/BaoLee956/TRAFFIC-DATA-SYSTEM.git
cd TRAFFIC-DATA-SYSTEM
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các gói thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi lệnh chạy thành công, trình duyệt web sẽ tự động mở giao diện tại:
👉 `http://localhost:8501`

---

## ⚡ Tối ưu hóa Kỹ thuật & Hiệu năng

1. **Caching với `@st.cache_data`**:
   - Áp dụng bộ nhớ đệm cho quá trình sinh/nạp dữ liệu và thuật toán DBSCAN. Khi người dùng thay đổi các bộ lọc ở sidebar, ứng dụng không phải chạy lại mô hình tốn tài nguyên.
2. **Zero-Rerun Folium Rendering**:
   - Sử dụng tham số `returned_objects=[]` trong `st_folium()` để loại bỏ việc truyền tín hiệu ngược từ bản đồ về máy chủ Streamlit khi người dùng cuộn/phóng to thu nhỏ bản đồ, đảm bảo độ mượt mà tối đa.
3. **MarkerCluster Optimization**:
   - Tối ưu hóa khả năng dựng hình cho hàng nghìn tọa độ GPS cùng lúc mà không làm nghẽn bộ nhớ trình duyệt client.

---

## 👨‍💻 Tác giả & Đóng góp

- **Tác giả**: [BaoLee956](https://github.com/BaoLee956)
- **Vai trò**: Senior Data Scientist & Streamlit Developer
- Mọi đóng góp, báo lỗi (Issues) hoặc yêu cầu tính năng mới (Pull Requests) đều được chào đón!

---

*Phát triển vì mục tiêu ứng dụng Trí tuệ Nhân tạo và Khoa học Dữ liệu trong Nâng cao An toàn Giao thông Đô thị.*
