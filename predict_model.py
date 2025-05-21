import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# JSON olarak gönderilen verileri al
input_json = sys.argv[1]
data = json.loads(input_json)

# Model ve encoder'ları yükle
def load_model_and_encoders():
    try:
        model = tf.keras.models.load_model('best_salary_prediction_model.h5')
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
            
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('y_scaler.pkl', 'rb') as f:
            y_scaler = pickle.load(f)
            
        return model, label_encoders, scaler, y_scaler
    except Exception as e:
        return None, None, None, None

# Tahmin yap
def predict(data):
    try:
        model, label_encoders, scaler, y_scaler = load_model_and_encoders()
        
        if not model or not label_encoders or not scaler or not y_scaler:
            raise Exception("Model dosyaları yüklenemedi")
        
        # Tahmin için girdiyi hazırla
        input_data = pd.DataFrame({
            'Cinsiyet': [data['cinsiyet']],
            'Şirketiniz hangi lokasyonda?': [data['lokasyon']],
            'Tam zamanlı mı yarı zamanlı mı çalışıyorsunuz?': [data['calisma_tipi']],
            'Çalışma şekliniz nedir?': [data['calisma_sekli']],
            'Şirketinizin çalışan sayısı nedir?': [data['sirket_buyuklugu']],
            'Kaç yıllık deneyiminiz var?': [data['deneyim']],
            'Çalıştığınız Sektor? (Otomotiv, Bankacilik, Saglik vs)': [data['sektor']],
            'Pozisyonunuzun Ismi? (Lütfen kısaltma kullanmadan yazınız)': [data['pozisyon']]
        })
        
        # Kategorik değişkenleri dönüştür
        categorical_columns = input_data.columns
        for column in categorical_columns:
            if column in label_encoders:
                input_data[column] = label_encoders[column].transform(input_data[column])
        
        # Veriyi ölçeklendir
        input_scaled = scaler.transform(input_data)
        
        # Tahmin yap
        prediction_scaled = model.predict(input_scaled)
        prediction = y_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Maaş aralığı hesapla (%15 alt ve üst sınır)
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        
        # Sonucu JSON formatında döndür
        result = {
            'success': True,
            'predicted_salary': float(prediction),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
        
        return result
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Tahmin sonucunu yazdır
result = predict(data)
print(json.dumps(result)) 