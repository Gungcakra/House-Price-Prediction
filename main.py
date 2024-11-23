import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('house_data.csv')

X = data[['size']]  
y = data['price'] 

#model
model = LinearRegression()

#train
model.fit(X, y)

# Menggunakan DataFrame untuk input prediksi, agar memiliki nama kolom yang sesuai
house_size = float(input("Masukkan ukuran rumah (m²): "))
input_data = pd.DataFrame([[house_size]], columns=['size'])

# Prediksi harga rumah berdasarkan ukuran
predicted_price = model.predict(input_data)

# Menampilkan hasil prediksi
print(f'Harga rumah dengan ukuran {house_size} m² diprediksi: Rp {predicted_price[0]:,.2f}')