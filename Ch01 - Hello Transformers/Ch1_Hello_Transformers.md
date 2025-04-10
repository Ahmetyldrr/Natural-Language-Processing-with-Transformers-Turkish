## The Encoder-Decoder Framework

## Encoder-Decoder (Kodlayıcı-Çözücü) Çerçevesi
Dönüştürücülerden (Transformers) önce, LSTMs gibi yinelenen (recurrent) mimariler Doğal Dil İşleme (NLP) alanında son teknoloji olarak kabul ediliyordu. Bu mimariler, ağ bağlantılarında bir geri besleme döngüsü içerir, bu da bilginin bir adımdan diğerine yayılmasına izin verir ve metin gibi sıralı verileri modellemek için idealdir.

## RNN (Yinelenen Sinir Ağları) Mimarisi
Bir RNN, bazı girdileri alır, bunları ağdan geçirir ve gizli durum (hidden state) adı verilen bir vektör çıktısı verir. Aynı zamanda, model, geri besleme döngüsü aracılığıyla kendisine bazı bilgiler geri besler, bu sayede bir sonraki adımda kullanabilir. Bu, döngüyü "açarsak" daha net bir şekilde görülebilir: RNN, her adımdaki durumu sıradaki bir sonraki işleme iletir. Bu, bir RNN'nin önceki adımlardaki bilgiyi takip etmesine ve çıktı tahminleri için kullanmasına olanak tanır.

## Encoder-Decoder (Kodlayıcı-Çözücü) Mimarisi
RNN'lerin önemli bir uygulama alanı, bir dildeki kelime dizisini başka bir dile çevirme görevi olan makine çeviri sistemlerinin geliştirilmesidir. Bu tür görevler genellikle encoder-decoder veya sequence-to-sequence mimarisi ile ele alınır. Kodlayıcının (encoder) görevi, girdi dizisindeki bilgiyi sayısal bir gösterime kodlamaktır, buna genellikle son gizli durum (last hidden state) denir. Bu durum daha sonra çıktı dizisini oluşturan çözücüye (decoder) iletilir.

## Önemli Noktalar:
* RNN'ler sıralı verileri modellemek için idealdir.
* Encoder-decoder mimarisi, girdi ve çıktı dizilerinin her ikisinin de keyfi uzunlukta olduğu durumlarda iyi uyum sağlar.
* Kodlayıcının son gizli durumu, tüm girdi dizisinin anlamını temsil etmek zorundadır, bu da bir bilgi darboğazı (information bottleneck) yaratır.
* Çözücünün tüm kodlayıcının gizli durumlarına erişmesine izin vererek bu darboğazdan kurtulmak mümkündür, buna dikkat mekanizması (attention mechanism) denir.

## Kod Örneği:
Aşağıdaki kod örneği, PyTorch kütüphanesini kullanarak basit bir encoder-decoder modelini göstermektedir.
```python
import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        return out[:, -1, :]

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.rnn(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Modeli tanımla
encoder = Encoder(input_dim=10, hidden_dim=20)
decoder = Decoder(hidden_dim=20, output_dim=10)

# Girdi verisini tanımla
x = torch.randn(1, 10, 10)

# Kodlayıcıdan geçir
encoded_x = encoder(x)

# Çözücüden geçir
output = decoder(encoded_x)

print(output.shape)
```
Bu kod, basit bir encoder-decoder modelini tanımlar ve girdi verisini bu modelden geçirir. Kodlayıcı, girdi dizisini gizli bir duruma kodlar ve bu durum daha sonra çözücü tarafından çıktı dizisini oluşturmak için kullanılır.

## Kodun Açıklaması:
* `Encoder` sınıfı, girdi dizisini gizli bir duruma kodlayan RNN'yi tanımlar.
* `Decoder` sınıfı, gizli durumu çıktı dizisine çeviren RNN'yi tanımlar.
* `forward` metodu, girdi verisini modelden geçirir ve çıktı verir.
* `h0` ve `c0` değişkenleri, LSTM hücresinin başlangıç durumlarını temsil eder.
* `out` değişkeni, RNN'nin çıktılarını temsil eder.
* `fc` katmanı, çıktıları istenen boyuta çevirmek için kullanılır.

## Dikkat Mekanizması (Attention Mechanism)
Dikkat mekanizması, çözücünün tüm kodlayıcının gizli durumlarına erişmesine izin vererek bilgi darboğazını aşmaya yardımcı olur. Bu, Transformer mimarisinin temel bileşenlerinden biridir.

---

## Attention Mechanisms

## Dikkat Mekanizmaları (Attention Mechanisms)

Dikkat mekanizmalarının arkasındaki ana fikir, girdi dizisi için tek bir gizli durum üretmek yerine, kodlayıcının (encoder) her adımda kod çözücü (decoder) tarafından erişilebilen bir gizli durum çıktısı vermesidir. Ancak, tüm durumları aynı anda kullanmak, kod çözücü için devasa bir girdi oluşturur, bu nedenle hangi durumların kullanılacağına öncelik vermek için bir mekanizmaya ihtiyaç vardır. İşte dikkat burada devreye girer: kod çözücünün her kod çözme adımında kodlayıcı durumlarının her birine farklı bir ağırlık veya "dikkat" atamasını sağlar. Bu süreç, Şekil 1-4'te gösterilmektedir.

## Dikkat Mekanizmalarının Çalışması

Dikkat mekanizmaları, her zaman adımında hangi girdi tokenlarının en alakalı olduğuna odaklanarak, oluşturulan bir çevirideki kelimeler ile kaynak cümledeki kelimeler arasında önemsiz olmayan hizalamaları öğrenebilir. Örneğin, Şekil 1-5, bir İngilizce-Fransızca çeviri modeli için dikkat ağırlıklarını görselleştirir, burada her piksel bir ağırlığı temsil eder.

## Transformer ve Self-Attention

Transformer ile yeni bir modelleme paradigması tanıtıldı: tekrarlayan (recurrent) yapıları tamamen bırakmak ve bunun yerine self-attention adı verilen özel bir dikkat biçimini kullanmak. Self-attention, dikkatin aynı katmandaki tüm durumlar üzerinde çalışmasına izin verir. Bu, Şekil 1-6'da gösterilmektedir.

## Kod Örneği

Aşağıdaki kod örneği, PyTorch kütüphanesini kullanarak bir dikkat mekanizmasının nasıl uygulanacağını gösterir:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, sequence_length, hidden_size)
        # decoder_hidden: (batch_size, hidden_size)
        encoder_outputs = self.W1(encoder_outputs)  # (batch_size, sequence_length, hidden_size)
        decoder_hidden = self.W2(decoder_hidden)  # (batch_size, hidden_size)
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
        scores = self.V(torch.tanh(encoder_outputs + decoder_hidden))  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, sequence_length, 1)
        return attention_weights

# Kullanımı
encoder_outputs = torch.randn(32, 10, 128)  # (batch_size, sequence_length, hidden_size)
decoder_hidden = torch.randn(32, 128)  # (batch_size, hidden_size)
attention = Attention(128)
attention_weights = attention(encoder_outputs, decoder_hidden)
print(attention_weights.shape)  # (32, 10, 1)
```
Bu kod, bir dikkat mekanizmasını tanımlar ve kullanır. `Attention` sınıfı, `encoder_outputs` ve `decoder_hidden` durumlarını alır ve dikkat ağırlıklarını hesaplar.

## Self-Attention Kod Örneği

Aşağıdaki kod örneği, PyTorch kütüphanesini kullanarak self-attention'ın nasıl uygulanacağını gösterir:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, hidden_size)
        Q = self.Q(x)  # (batch_size, sequence_length, hidden_size)
        K = self.K(x)  # (batch_size, sequence_length, hidden_size)
        V = self.V(x)  # (batch_size, sequence_length, hidden_size)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_size)  # (batch_size, sequence_length, sequence_length)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, sequence_length, sequence_length)
        output = torch.matmul(attention_weights, V)  # (batch_size, sequence_length, hidden_size)
        return output

# Kullanımı
x = torch.randn(32, 10, 128)  # (batch_size, sequence_length, hidden_size)
self_attention = SelfAttention(128)
output = self_attention(x)
print(output.shape)  # (32, 10, 128)
```
Bu kod, self-attention'ı tanımlar ve kullanır. `SelfAttention` sınıfı, `x` durumunu alır ve self-attention'ı uygular.

## Transfer Learning

Transformer devriminin başlaması için son bir parça eksikti: transfer learning. Transfer learning, önceden eğitilmiş bir modelin başka bir görev için kullanılmasını sağlar. Bu, NLP görevlerinde büyük başarılar elde edilmesini sağlamıştır.

---

## Transfer Learning in NLP

## Doğal Dil İşlemede (NLP) Transfer Öğrenme

Görüntü işleme alanında, bir Convolutional Neural Network (CNN) gibi ResNet'i bir görevde eğitmek ve daha sonra yeni bir göreve uyarlamak veya ince ayar yapmak için transfer öğrenme kullanmak günümüzde yaygın bir uygulamadır. Bu, ağın orijinal görevden öğrenilen bilgiyi kullanmasına olanak tanır. Mimari olarak, bu, modeli bir gövde ve bir başlık olarak ayırmayı içerir, burada başlık görev-specifik bir ağdır. Eğitim sırasında, gövdenin ağırlıkları kaynak alanının geniş özelliklerini öğrenir ve bu ağırlıklar yeni görev için yeni bir modeli başlatmak için kullanılır.

## Transfer Öğrenmenin Avantajları
* Yüksek kaliteli modeller üretir
* Çeşitli downstream görevlerinde daha verimli bir şekilde eğitilebilir
* Daha az etiketli veri gerektirir

## Doğal Dil İşlemede (NLP) Transfer Öğrenme

Görüntü işlemede, modeller ilk olarak ImageNet gibi büyük ölçekli veri kümelerinde eğitilir. Bu süreç, ön eğitim (pretraining) olarak adlandırılır ve temel amacı, modellere görüntülerin temel özelliklerini, kenarları veya renkleri öğretmektir. Bu önceden eğitilmiş modeller daha sonra nispeten az sayıda etiketli örnekle (genellikle sınıf başına birkaç yüz) çiçek türlerini sınıflandırmak gibi bir downstream görevinde ince ayar yapılabilir. İnce ayar yapılmış modeller, aynı miktarda etiketli veri üzerinde sıfırdan eğitilen denetimli modellerden daha yüksek bir doğruluk elde eder.

## NLP'de Transfer Öğrenmenin Gelişimi

2017 ve 2018'de, birkaç araştırma grubu, NLP için transfer öğrenmeyi işe yarayan yeni yaklaşımlar önerdi. OpenAI'deki araştırmacılar, duygu sınıflandırma görevinde güçlü performans elde ederek, denetimsiz ön eğitimden çıkarılan özellikler kullanarak bu süreci başlattılar. Bunu, önceden eğitilmiş LSTM modellerini çeşitli görevlere uyarlamak için genel bir çerçeve tanıtan ULMFiT izledi.

## ULMFiT'in Adımları

1. **Dil Modelleme (Language Modeling)**: İlk eğitim amacı oldukça basit: önceki kelimelere dayanarak sonraki kelimeyi tahmin etmek.
2. **Alan içi (In-Domain) İnce Ayar**: Önceden eğitilmiş dil modeli, hedef veri kümesine (örneğin, Wikipedia'dan IMDb film eleştirileri veri kümesine) ince ayar yapılır.
3. **Sınıflandırma Katmanı Ekleme**: Dil modeli, hedef görev için bir sınıflandırma katmanı ile ince ayar yapılır (örneğin, film eleştirilerinin duygu sınıflandırması).

## ULMFiT'in Kod Örneği
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

# Önceden eğitilmiş model ve tokenizer yükleme
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Veri kümesini yükleme ve tokenize etme
train_data = ...  # Veri kümesini yükleme
train_encodings = tokenizer(train_data["text"], truncation=True, padding=True)

# Dil modelini ince ayar yapma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_encodings:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_encodings)}")
```
Bu kod, önceden eğitilmiş bir DistilBERT modelini yükler ve bir duygu sınıflandırma görevinde ince ayar yapar.

## GPT ve BERT

2018'de, GPT ve BERT adlı iki transformer modeli yayınlandı. GPT, sadece decoder kısmını kullanırken, BERT encoder kısmını kullanır. GPT, BookCorpus üzerinde önceden eğitildi, BERT ise BookCorpus ve İngilizce Wikipedia üzerinde önceden eğitildi. Her iki model de çeşitli NLP kıyaslamalarında yeni bir state-of-the-art elde etti.

## Transformers Kütüphanesi

Transformers kütüphanesi, 50'den fazla mimari için birleşik bir API sağlar. Bu kütüphane, transformer araştırmalarının patlamasına katalizör oldu ve kısa sürede NLP uygulayıcılarına ulaştı, bu modelleri birçok gerçek dünya uygulamasına entegre etmeyi kolaylaştırdı.

---

## Hugging Face Transformers: Bridging the Gap

## Hugging Face Transformers: Doğal Dil İşleme (NLP) Görevlerinde Kolaylık Sağlama

Yeni bir makine öğrenimi (Machine Learning) mimarisini (architecture) yeni bir görev için uygulamak karmaşık bir süreçtir ve genellikle aşağıdaki adımları içerir: 
- Model mimarisini (model architecture) kod olarak uygulamak, genellikle PyTorch veya TensorFlow kullanılarak yapılır.
- Eğitilmiş ağırlıkları (pretrained weights) bir sunucudan yüklemek.
- Girdileri ön işlemden geçirmek (preprocess), model üzerinden geçirmek ve görev-specifik (task-specific) son işlemleri uygulamak.
- Veri yükleyicileri (dataloaders) uygulamak ve modelin eğitimi için kayıp fonksiyonları (loss functions) ve optimizasyon algoritmaları (optimizers) tanımlamak.

## Hugging Face Transformers Kütüphanesinin Sağladığı Kolaylıklar

Hugging Face Transformers kütüphanesi, çeşitli transformer modellerine standart bir arayüz (interface) sağlar ve bu modelleri yeni kullanım durumlarına (use cases) uyarlamak için kod ve araçlar sunar. 
- Şu anda üç büyük derin öğrenme (deep learning) çerçevesini (PyTorch, TensorFlow ve JAX) desteklemektedir ve aralarında kolayca geçiş yapmanıza olanak tanır.
- Metin sınıflandırma (text classification), adlandırılmış varlık tanıma (named entity recognition) ve soru cevaplama (question answering) gibi downstream görevlerde transformer modellerini kolayca fine-tune etmenizi sağlayan görev-specifik başlıklar (task-specific heads) sağlar.

## Örnek Kod

Aşağıdaki örnek kodda, Hugging Face Transformers kütüphanesini kullanarak bir metin sınıflandırma görevi için nasıl bir modelin fine-tune edileceği gösterilmektedir:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Veri setini yükleme
df = pd.read_csv("your_data.csv")

# Veriyi eğitim ve test setlerine ayırma
train_text, val_text, train_labels, val_labels = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Tokenizer ve modeli yükleme
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)

# Özel veri seti sınıfı tanımlama
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Veri seti ve veri yükleyicileri oluşturma
train_dataset = TextDataset(train_text, train_labels, tokenizer)
val_dataset = TextDataset(val_text, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Modeli fine-tune etme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / len(val_labels)
        print(f"Epoch {epoch+1}, Val Acc: {accuracy:.4f}")
```
Bu kod, `distilbert-base-uncased` modelini metin sınıflandırma görevi için fine-tune etmektedir. 
- `AutoModelForSequenceClassification` ve `AutoTokenizer` sınıfları, Hugging Face Transformers kütüphanesinden yüklenir.
- Veri seti, `train_test_split` fonksiyonu kullanılarak eğitim ve test setlerine ayrılır.
- `TextDataset` sınıfı, özel veri seti sınıfı olarak tanımlanır.
- `DataLoader` sınıfı, veri yükleyicileri oluşturmak için kullanılır.
- Model, `torch.nn.CrossEntropyLoss()` kayıp fonksiyonu ve `torch.optim.Adam` optimizasyon algoritması kullanılarak fine-tune edilir.

## Kodun Açıklaması

- `from transformers import AutoModelForSequenceClassification, AutoTokenizer`: Hugging Face Transformers kütüphanesinden `AutoModelForSequenceClassification` ve `AutoTokenizer` sınıflarını yükler.
- `tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")`: `distilbert-base-uncased` modelinin tokenizer'ını yükler.
- `model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=8)`: `distilbert-base-uncased` modelini metin sınıflandırma görevi için yükler.
- `TextDataset` sınıfı: Özel veri seti sınıfı olarak tanımlanır. Bu sınıf, metin verilerini ve etiketleri içerir.
- `DataLoader` sınıfı: Veri yükleyicileri oluşturmak için kullanılır.
- `criterion = torch.nn.CrossEntropyLoss()`: Kayıp fonksiyonu olarak `CrossEntropyLoss` kullanılır.
- `optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)`: Optimizasyon algoritması olarak `Adam` kullanılır.

## Sonuç

Hugging Face Transformers kütüphanesi, çeşitli transformer modellerine standart bir arayüz sağlar ve bu modelleri yeni kullanım durumlarına uyarlamak için kod ve araçlar sunar. Bu kütüphane, metin sınıflandırma, adlandırılmış varlık tanıma ve soru cevaplama gibi downstream görevlerde transformer modellerini kolayca fine-tune etmenizi sağlar. Yukarıdaki örnek kod, bu kütüphanenin nasıl kullanılacağını göstermektedir.

---

## A Tour of Transformer Applications

## Transformer Uygulamalarına Bir Bakış
Her bir NLP görevi, aşağıdaki gibi bir metin parçasıyla başlar: 
müşteri geri bildirimleri (customer feedback) 
örnek metin: 
text = """Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
Uygulamanıza bağlı olarak, üzerinde çalıştığınız metin bir yasal sözleşme, ürün açıklaması veya tamamen başka bir şey olabilir.

## Duygu Analizi (Sentiment Analysis)
Müşteri geri bildirimlerinde duygu analizi yapmak isteyebilirsiniz. Bu görev, metin sınıflandırma (text classification) olarak bilinir. 
Transformers kütüphanesini kullanarak duygu analizini gerçekleştirmek için:
```python
from transformers import pipeline
classifier = pipeline("text-classification")
```
Bu kod, otomatik olarak Hugging Face Hub'dan model ağırlıklarını indirir.

## Tahminlerin Oluşturulması
Her bir pipeline, girdi olarak bir metin dizisi (veya metin dizilerinin listesi) alır ve bir tahminler listesi döndürür. 
Her bir tahmin bir Python sözlüğüdür, bu nedenle Pandas kullanarak bunları güzel bir şekilde DataFrame olarak görüntüleyebiliriz:
```python
import pandas as pd
outputs = classifier(text)
pd.DataFrame(outputs)
```
Bu durumda model, metnin negatif bir duygu içerdiğinden çok emin.

## Adlandırılmış Varlık Tanıma (Named Entity Recognition, NER)
Müşteri geri bildirimlerindeki adlandırılmış varlıkları tanımak isteyebilirsiniz. 
Bunu yapmak için:
```python
ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)
```
Bu pipeline, metindeki varlıkları tespit eder ve her birine bir kategori atar (örneğin, ORG, LOC, PER).

## Soru-Cevap (Question-Answering)
Metinden belirli bir soruya cevap bulmak isteyebilirsiniz. 
Bunu yapmak için:
```python
reader = pipeline("question-answering")
question = "What does the customer want?"
outputs = reader(question=question, context=text)
pd.DataFrame([outputs])
```
Bu pipeline, soruya bir cevap verir ve cevabın metindeki konumunu döndürür.

## Özetleme (Summarization)
Uzun bir metni özetlemek isteyebilirsiniz. 
Bunu yapmak için:
```python
summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])
```
Bu pipeline, metnin özetini verir.

## Çeviri (Translation)
Bir metni başka bir dile çevirmek isteyebilirsiniz. 
Bunu yapmak için:
```python
translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])
```
Bu pipeline, metni belirtilen dile çevirir.

## Metin Oluşturma (Text-Generation)
Müşteri geri bildirimlerine otomatik cevaplar oluşturmak isteyebilirsiniz. 
Bunu yapmak için:
```python
generator = pipeline("text-generation")
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])
```
Bu pipeline, verilen metne göre bir cevap oluşturur.

## Önemli Noktalar
* Transformers kütüphanesi, NLP görevleri için çeşitli pipeline'lar sağlar.
* Pipeline'lar, metinleri işlemek ve tahminler oluşturmak için kullanılır.
* Her bir pipeline, belirli bir görev için eğitilmiş bir model kullanır.
* Modeller, Hugging Face Hub'dan indirilebilir.
* Pipeline'lar, çeşitli parametrelerle özelleştirilebilir. 

### Kullanılan Kodların Açıklaması
* `pipeline()` fonksiyonu, bir pipeline oluşturmak için kullanılır.
* `classifier`, `ner_tagger`, `reader`, `summarizer`, `translator`, `generator` pipeline'ları, sırasıyla duygu analizi, adlandırılmış varlık tanıma, soru-cevap, özetleme, çeviri ve metin oluşturma görevleri için kullanılır.
* `text` değişkeni, örnek metni içerir.
* `outputs` değişkeni, pipeline'dan dönen tahminleri içerir.
* `pd.DataFrame()` fonksiyonu, tahminleri bir DataFrame olarak görüntülemek için kullanılır.
* `print()` fonksiyonu, özet, çeviri ve metin oluşturma sonuçlarını görüntülemek için kullanılır. 

### Kullanılan Kütüphaneler
* `transformers`
* `pandas` 

### İçe Aktarılan Modüller
```python
from transformers import pipeline
import pandas as pd
```

---

## The Hugging Face Ecosystem

## Hugging Face Ekosistemi
Hugging Face ekosistemi, NLP (Doğal Dil İşleme) ve makine öğrenimi projelerinizi hızlandırmak için birçok kütüphane ve araçtan oluşur. Ekosistemin ana iki bölümü vardır: bir kütüphane ailesi ve Hub. Kütüphaneler kodu sağlarken, Hub önceden eğitilmiş model ağırlıkları, veri kümeleri, değerlendirme metrikleri için komut dosyaları ve daha fazlasını sağlar.

## Ekosistemin Bileşenleri
Ekosistemin bileşenleri şunlardır:
* Transformers: Daha önce tartıştığımız gibi, transfer öğrenimi transformer'ların başarısının anahtar faktörlerinden biridir. 
* Hub: Hugging Face Hub, 20.000'den fazla ücretsiz model barındırır. 
* Tokenizers: Tokenizers, metni daha küçük parçalara ayırarak token'lara dönüştürür. 
* Datasets: Datasets, veri kümelerini yükleme, işleme ve depolama işlemlerini basitleştirir. 
* Accelerate: Accelerate, eğitim döngüsü üzerinde ince taneli kontrol sağlar.

## Hub
Hub, önceden eğitilmiş model ağırlıkları, veri kümeleri ve değerlendirme metrikleri için komut dosyaları sağlar. 
```python
from transformers import pipeline

# Modeli yükleme
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name)
```
Yukarıdaki kodda, `pipeline` fonksiyonu kullanılarak önceden eğitilmiş bir model yüklenir. Model adı `distilbert-base-uncased-finetuned-sst-2-english` olarak belirlenmiştir.

## Tokenizers
Tokenizers, metni token'lara dönüştürür. 
```python
from tokenizers import Tokenizer

# Tokenizer yükleme
tokenizer = Tokenizer.from_file("path/to/tokenizer.json")

# Metni token'lara dönüştürme
output = tokenizer.encode("This is a sample sentence.")
print(output.tokens())
```
Yukarıdaki kodda, `Tokenizer` sınıfı kullanılarak bir tokenizer yüklenir ve metni token'lara dönüştürür.

## Datasets
Datasets, veri kümelerini yükleme, işleme ve depolama işlemlerini basitleştirir. 
```python
from datasets import load_dataset

# Veri kümesini yükleme
dataset = load_dataset("glue", "sst2")

# Veri kümesini işleme
dataset = dataset.map(lambda examples: {"text": [example["sentence"] for example in examples]}, batched=True)
```
Yukarıdaki kodda, `load_dataset` fonksiyonu kullanılarak bir veri kümesi yüklenir ve `map` fonksiyonu kullanılarak veri kümesi işlenir.

## Accelerate
Accelerate, eğitim döngüsü üzerinde ince taneli kontrol sağlar. 
```python
import torch
from accelerate import Accelerator

# Accelerator oluşturma
accelerator = Accelerator()

# Modeli ve verileri hazırlama
model = torch.nn.Module()
dataloader = torch.utils.data.DataLoader(dataset)

# Eğitimi hızlandırma
model, dataloader = accelerator.prepare(model, dataloader)

# Eğitim döngüsü
for batch in dataloader:
    # Eğitim adımları
    pass
```
Yukarıdaki kodda, `Accelerator` sınıfı kullanılarak bir accelerator oluşturulur ve eğitim döngüsü hızlandırılır.

## Sonuç
Hugging Face ekosistemi, NLP ve makine öğrenimi projelerinizi hızlandırmak için birçok kütüphane ve araç sağlar. Ekosistemin bileşenleri, Transformers, Hub, Tokenizers, Datasets ve Accelerate'dir. Bu kütüphaneler, önceden eğitilmiş model ağırlıkları, veri kümeleri, tokenization ve eğitim döngüsü üzerinde ince taneli kontrol sağlar.

---

## Main Challenges with Transformers

## Transformatörlerle İlgili Ana Zorluklar (Main Challenges with Transformers)

Transformatör modelleri (Transformer Models) birçok Doğal Dil İşleme (NLP) görevinde başarılı bir şekilde kullanılmaktadır. Ancak, bu modellerin bazı zorlukları da vardır. Bu bölümde, transformatör modellerinin karşılaştığı bazı zorlukları ele alacağız.

## Transformatör Modellerinin Karşılaştığı Zorluklar

*   İngilizce dilinin hakim olduğu NLP araştırmalarında, diğer diller için önceden eğitilmiş (pre-trained) modeller bulmak zor olabilir. Bu sorun, özellikle nadir veya düşük kaynaklı diller (low-resource languages) için geçerlidir. 
*   Aktarım öğrenme (Transfer Learning) sayesinde, modellerin ihtiyaç duyduğu etiketli eğitim verilerinin (labeled training data) miktarı azaltılabilir, ancak yine de bir insanın görevi yerine getirmek için ihtiyaç duyduğu veri miktarına kıyasla çok fazladır. 
*   Kendine dikkat (Self-Attention) mekanizması, paragraf uzunluğundaki metinlerde çok iyi çalışır, ancak daha uzun metinlerde, örneğin tüm belgelerde (whole documents), çok pahalı hale gelir. 
*   Diğer derin öğrenme modelleri (deep learning models) gibi, transformatör modelleri de büyük ölçüde opak (opaque) olabilir. Bir modelin belirli bir tahminde bulunmasının nedenini çözmek zor veya imkansız olabilir. 
*   Transformatör modelleri ağırlıklı olarak internetten alınan metin verileri (text data) üzerinde önceden eğitilir. Bu, verilerdeki tüm önyargıları (biases) modellere aktarır. Bu önyargıların ne ırkçı (racist) ne de cinsiyetçi (sexist) olmamasını sağlamak zor bir görevdir.

## Örnek Kod Parçaları ve Açıklamaları

Aşağıdaki kod parçaları, transformatör modellerinin kullanımına örnek teşkil etmektedir.

### Transformers Kütüphanesini İçe Aktarma (Importing Transformers Library)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
```

Bu kod, `transformers` kütüphanesinden `AutoModelForSequenceClassification` ve `AutoTokenizer` sınıflarını içe aktarır. `AutoModelForSequenceClassification`, otomatik olarak bir dizi sınıflandırma görevi için uygun modeli seçer ve `AutoTokenizer`, modele uygun tokenleştiriciyi (tokenizer) seçer.

### Model ve Tokenizeri Yükleme (Loading Model and Tokenizer)

```python
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Bu kod, `distilbert-base-uncased-finetuned-sst-2-english` adlı önceden eğitilmiş modeli ve tokenleştiriciyi yükler.

### Metni Tokenleştirme (Tokenizing Text)

```python
text = "This is an example sentence."
inputs = tokenizer(text, return_tensors="pt")
```

Bu kod, `text` değişkenindeki metni tokenleştirir ve `inputs` değişkenine atar. `return_tensors="pt"` parametresi, çıktıların PyTorch tensörleri (tensors) olarak döndürülmesini sağlar.

### Model Çıkışını Alma (Getting Model Output)

```python
outputs = model(**inputs)
```

Bu kod, `inputs` değişkenindeki tokenleştirilmiş metni modele verir ve model çıkışını `outputs` değişkenine atar.

Bu örnek kod parçaları, transformatör modellerinin kullanımına basit bir giriş sağlar. Daha fazla bilgi ve detaylı örnekler için ilgili kaynaklara başvurabilirsiniz.

---

## Conclusion

## Sonuç (Conclusion)
Umarız ki artık bu çok yönlü modelleri kendi uygulamalarınıza entegre etmek ve eğitmeye başlamak için heyecan duyuyorsunuzdur! (Hopefully, by now you are excited to learn how to start training and integrating these versatile models into your own applications!)

## Önemli Noktalar (Key Points)
* Bu bölümde, sadece birkaç satır kod (code) ile sınıflandırma (classification), adlandırılmış varlık tanıma (named entity recognition), soru cevaplama (question answering), çeviri (translation) ve özetleme (summarization) için son teknoloji (state-of-the-art) modelleri kullanabileceğinizi gördünüz.
* Bu, gerçekten sadece "buzdağının görünen kısmı"dır (tip of the iceberg).
* Sonraki bölümlerde, bir metin sınıflandırıcı (text classifier) oluşturma, üretim için hafif bir model (lightweight model) oluşturma veya sıfırdan bir dil modeli (language model) eğitme gibi geniş bir kullanım alanına (wide range of use cases) transformatorları nasıl adapte edeceğinizi öğreneceksiniz.
* Uygulamalı bir yaklaşım izlenecektir, yani her bir konu için Google Colab veya kendi GPU makinenizde çalıştırabileceğiniz eşlik eden kodlar (accompanying code) olacaktır.

## Kullanılan Kodlar (Used Codes)
Bu metinde spesifik bir kod örneği verilmemektedir, ancak sonraki bölümlerde kullanılacak kodların Google Colab veya kendi GPU makinenizde çalıştırılabileceği belirtilmektedir. Örneğin, bir transformator modelini yüklemek ve kullanmak için aşağıdaki kod bloğu kullanılabilir:
```python
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Model ve tokenizer yükleme
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Metin sınıflandırma için örnek kod
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    return torch.argmax(logits)

# Örnek kullanım
text = "Bu bir örnek metindir."
print(classify_text(text))
```
## Kod Açıklaması (Code Explanation)
Yukarıdaki kod bloğunda:
* `import` ifadeleri ile gerekli kütüphaneler yüklenmektedir. (`pandas`, `torch`, `transformers`)
* `AutoModelForSequenceClassification` ve `AutoTokenizer` sınıflarını kullanarak bir transformator modelini ve tokenizer'ı yüklüyoruz.
* `classify_text` fonksiyonu, verilen bir metni sınıflandırmak için kullanılmaktadır.
* `tokenizer` ile metin tokenlara ayrılmakta ve modele girdi olarak hazırlanmaktadır.
* `model` ile sınıflandırma işlemi gerçekleştirilmekte ve sonuç döndürülmektedir.

Bu kod bloğu, bir metin sınıflandırma görevi için transformator modellerini nasıl kullanabileceğinizi göstermektedir.

---

