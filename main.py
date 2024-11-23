import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('house_data.csv')

X = data[['size']]  
y = data['price'] 

# Model
model = LinearRegression()

# Latih model
model.fit(X, y)

# Menggunakan DataFrame untuk input prediksi, agar memiliki nama kolom yang sesuai
house_size = float(input("Input house size (m²): "))
input_data = pd.DataFrame([[house_size]], columns=['size'])

# Prediksi harga rumah berdasarkan ukuran
predicted_price = model.predict(input_data)

# Menampilkan hasil prediksi
print(f'The predicted price of a house with a size of {house_size} m² is: Rp {predicted_price[0]:,.2f}')
