# Simulation of Multi-Armed Bandits (MAB) in Buybox system

- **Bài toán MAB**: Cho trước một tập hợp các sự lựa chọn, mỗi sự lựa chọn có một xác suất thành công không biết trước. Ở mỗi lượt, một agent cần phải lựa chọn và nhận kết quả (thành công hoặc thất bại). Hãy tìm một chiến thuật tối đa hoá thành công sau T lượt.

- **Thuật toán**: Các thuật toán giải MAB đi tìm sự cân bằng giữa chiến thuật khai thác (Exploitation) và khai phá (Exploration). Khai thác là tận dụng kinh nghiệm thu thập được trong quá khứ để quyết định (xem lựa chọn mang lại thành công nhiều nhất cho đến thời điểm hiện tại là lựa chọn tối ưu). Ngược lại, khai phá khuyến khích những lựa chọn ít được chọn nhằm làm giảm khả năng bỏ sót các lựa chọn tối ưu.

## Thuật toán không sử dụng ngữ cảnh (context)
Các thuật toán này chỉ sử dụng thống kê về kết quả sau các lần thí nghiệm, chính vì vậy phù hợp với các bài toán trong đó phần thưởng không bị ảnh hưởng bởi các yếu tố bên ngoài. Ví dụ: trong bài toán máy đánh bạc, mỗi máy có 1 xác suất thắng/thua cố định và không bị thay đổi bởi người chơi.

## Thuật toán sử dụng ngữ cảnh
Các thuật toán này sử dụng cả thống kê về kết quả và các yếu tố có thể ảnh hưởng đến kết quả. Ví dụ: trong bài toán Buybox, người dùng có những sở thích mua hàng riêng biệt và ảnh hưởng đến việc chọn offer tốt nhất.

## Simulation
Repo này cài đặt một số thuật toán:

- Không sử dụng ngữ cảnh: Epsilon-Greedy, Upper Confidence Bound (UCB)
- Sử dụng ngữ cảnh: Linear UCB (LinUCB)

Đối với thuật toán LinUCB, ngữ cảnh được giả lập như sau:

- Mỗi khách hàng có sở thích mua hàng giá rẻ và giao nhanh khác nhau, nhận giá trị trong `[0, 1]`. Các giá trị này là các biến ngẫu nhiên, phân bố Gaussian với mean lần lượt là 0.7 và 0.6, variance 0.01.

```python
cheap = np.clip(0.1*np.random.randn() + 0.7, 0, 1)
fast  = np.clip(0.1*np.random.randn() + 0.6, 0, 1)
```

- Mỗi sản phẩm có hai thuộc tính `best_price` và `best_pdd`, nhận giá trị nhị phân `0` hoặc `1`. Để đơn giản, giả định một buybox có 3 offers:

```
OfferContext(best_price=0, best_pdd=1)
OfferContext(best_price=0, best_pdd=0)
OfferContext(best_price=1, best_pdd=0)
```

- Phần thưởng cho sự chọn lựa offer O cho customer C sẽ là 1 với xác suất 

```
(C.cheap * O.best_price + C.fast * O.best_pdd ) / 2
```

và là 0 với xác suất:

```
1 - (C.cheap * O.best_price + C.fast * O.best_pdd ) / 2
```

## Kết quả:
Thực thi:

```
pip install -r requirements.txt
python sim.py
```

Số lượng phần thưởng tích lũy của các thuật toán sau `T = 1000` vòng lặp:
![GitHub Logo](/assets/algo_comparison.png)

Lịch sử log của thuật toán LinUCB.
![GitHub Logo](/assets/linucb_details.png)