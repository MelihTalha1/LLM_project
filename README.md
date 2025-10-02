# Mini-LLM
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KSeQbTLw7rKPTSIyHQVQA063aWUr4v08?usp=sharing)

<img width="628" height="458" alt="image" src="https://github.com/user-attachments/assets/8b03fbf6-e79a-4876-8b3c-638076bc1db1" />


Bu proje, hafif bir Büyük Dil Modeli (LLM) için geliştirilmiş bir Gradio tabanlı web arayüzüdür. Amacı, sıfırdan oluşturulan Transformer tabanlı dil modellerini kolayca yüklemek, test etmek ve çeşitli çıkarım parametreleri (Temperature, Top-P, vb.) ile etkileşimli olarak denemektir.

## Metin Üretimini Kontrol Etme
Sohbet alanının solunda yer alan kaydırıcılar ve giriş kutuları ile modelin çıktılarını hassas bir şekilde ayarlayabilirsiniz:

| Ayar         | Açıklama                                             | Deneyim                            |
|--------------|-----------------------------------------------------|----------------------------------|
| Max Tokens   | Üretilecek maksimum belirteç sayısı. |Yanıtın uzunluğunu sınırlar. |
| Temperature  | Raslantısallık seviyesi. Düşük (0.1): Tutucu, en olası yanıtlara odaklanır. Yüksek (0.9): Yaratıcı, deneysel ve çeşitli yanıtlar üretir. | Düşük (0.1): Tutucu, en olası yanıtlara odaklanır. Yüksek (0.9): Yaratıcı, deneysel ve çeşitli yanıtlar üretir.|
| Top-K / Top-P| Örnekleme stratejileri.|Çıktıdaki olası kelime dağarcığını daraltarak daha alakalı veya odaklanmış sonuçlar sağlar.                                  |
