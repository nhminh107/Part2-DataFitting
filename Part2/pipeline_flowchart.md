# Sơ đồ Luồng xử lý DataPipeline

Sơ đồ được thiết kế với **2 hàng ngang** (Sử dụng `flowchart` để đảm bảo hiển thị đúng bố cục trên các trình xem hiện đại).

```mermaid
flowchart TD
    %% Styling
    classDef init fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:1px;
    classDef subProcess fill:#d4edda,stroke:#333,stroke-width:1px;
    classDef output fill:#ff9,stroke:#333,stroke-width:2px;

    %% HÀNG 1
    subgraph FIT ["GIAI ĐOẠN 1: FIT (Học tham số)"]
        direction LR
        Start(["Dữ liệu thô"]):::init --> P1("1. Chuẩn bị &<br/>Xử lý Target")
        P1 --> P2("2. Phân loại cột<br/>Num/Cat")
        P2 --> P3("3. Học Impute<br/>KNN/MICE")
        P3 --> P4("4. Học Outlier &<br/>Scaling")
        P4 --> P5("5. Học khuôn<br/>Encoding")
    end

    %% Mối nối dọc để chuyển hàng
    FIT ==> TRANSFORM

    %% HÀNG 2
    subgraph TRANSFORM ["GIAI ĐOẠN 2: TRANSFORM (Biến đổi)"]
        direction LR
        T1("1. Điền<br/>Missing Value") --> T2("2. Cắt Outlier<br/>Winsorize")
        T2 --> T3("3. Mã hóa<br/>Ord & Nom")
        T3 --> T4("4. Chuẩn hóa<br/>Z-Score") --> End(["Dữ liệu sạch"]):::output
    end
```

### Tại sao sơ đồ này hiển thị tốt trong LaTeX?
1.  **Cấu trúc 2 hàng**: Thay vì một dải dài dằng dặc, sơ đồ được chia đôi. `Giai đoạn 1` ở trên, `Giai đoạn 2` ở dưới.
2.  **Sử dụng `flowchart`**: Đây là phiên bản mới hơn của `graph`, hỗ trợ tốt hơn việc quy định hướng (`direction LR`) bên trong từng khối (subgraph).
3.  **Mũi tên nối dày (`==>`)**: Tạo ra sự phân cấp rõ ràng giữa việc hoàn tất quá trình "Học" và bắt đầu quá trình "Biến đổi".
4.  **Bố cục cân đối**: Giúp hình ảnh khi chèn vào LaTeX không bị co quá nhỏ chiều ngang, giữ cho chữ luôn dễ đọc.
