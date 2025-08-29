#Proje nasıl çalışıtırılır
python -m uvicorn script:app --reload

#Bitirme proje genel kütüphaneler kullanmışlar:
1- re 
2- torch 
3- emoji
4- numpy
5- sklearn
6- transformers
7- fastapi
8- String

#Bu proje kullanmak için bazı kütüphaneler indirmem lazım! 
✅ FastAPI kütüphaneleri – web uygulaması oluşturmak için:
                                        Kütüphane           kurulumu
1-Oluşturmak                            fastapi             ✅ pip install fastapi
2-Verileri iletmek için                 Request 	        fastapi içinde
3-Kullanıcının bir HTML                 Form                fastapi içinde
formu aracılığıyla 
girdiği verileri almak için  
4-Kullanıcı dosya yüklemelerini 
(metin dosyaları gibi) işlemek için     UploadFile, File    fastapi içinde


🔹2. Jinja2Templates – Sonuçları HTML sayfalarında görüntülemek için:
                                        Kütüphane                               kurulumu
1-Jinja2 kullanarak Python 
verilerini HTML sayfalarına bağlamak    fastapi.templating.Jinja2Templates      FastAPI'ye entegre edildi


🔹3. Regex ve string – metinleri temizlemek için:
                                        Kütüphane              kurulumu
1-Düzenli ifadeleri kullanarak          re                     (Python'a dahil)
sembolleri ve etiketleri 
kaldırmak için kullanılır.
2-Metinden kaldırılacak noktalama 
işaretlerinin bir listesini içerir.     string	               (Python'a dahil)


🔹4. emoji – Metinlerden emojileri kaldırmak için:
| Kütüphane | Kurulum | ✅ pip emojiyi yükle | Emojileri metinden güvenli ve hızlı bir şekilde kaldırmak için kullanılır |

🔹5. PyTorch – Model işleme için:
| Kütüphane | Kurulum | ✅ pip meşaleyi kur | Yeni eğitim olmadan modeli yüklemek ve tahminler üretmek için kullanılır |

🔹6. Transformers – BERT modelini indirmek ve kullanmak için:
| Kütüphane | Kurulum | ✅ pip trafo kurulumu | HuggingFace kütüphanesinden, BERT modelini yüklemek ve duyguyu analiz etmek için kullanılır | 
| Kullanılan malzemeler: | | - BertTokenizer | Metni formdan anlaşılan Token'lara dönüştürmek için | | - BertForSequenceSınıflandırması | Metin sınıflandırması için önceden eğitilmiş BERT modeli (duygu analizi) |

🔹7. NumPy – TF-IDF verilerini işlemek için:
| Kütüphane | Kurulum | ✅ pip numpy'ı kurun | Özellik matrislerini liste haline getirip değerleri düzenli bir şekilde görüntülemek için kullanılır.

🔹 8. Scikit-learn – TF-IDF özelliklerini hesaplamak için:
| Kütüphane | Kurulum | ✅ pip install scikit-learn | TfidfVectorizer kullanılarak metinlerdeki kelimelerin önemini hesaplamak için kullanılır |





