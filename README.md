# Skimmias-Defects-Segmentation

Projekt polega na stworzeniu modelu wykrywającego różne defekty kwiatów Skimmi Rubelli. Projekt zawiera analizę cech charakterystycznych poszczególnych defektów kwiatków w celu znalezienia odpowiedniego preprocessingu. Analiza będzie bezpośrednio wpływać na wybór odpowiedniego modelu oraz jego efektywność. Projekt zawiera implementację modeli segmentacyjncyh FCN (Fully Convolutional Neural Network) oraz modelu segmentacyjnego U-NET. Wykrycie poszczególnych uszkodzeń kwiatów pozwoli na odpowiednią reakcję i zapobieganie rozprzestrzeniania się problemu na większą skalę. Takie działanie zapobiegawcze może uchronić firmę przed większymi szkodami finansowymi. 

| Pogryzienie | Przypalenie |
|-------------|-------------|
|  ![Image_39](https://github.com/user-attachments/assets/e37b159c-cbf8-4f09-9050-8506f4aa575d) | ![Image_151](https://github.com/user-attachments/assets/9833d3c9-0178-4972-932f-817e70376b8d) |

Zbiór danych składa się z ponad 300 zdjęć, które zawierają roślny wraz z ich defektami tj. pogryzienia i przypalenia oraz odpowiadające im maski. Zbiór danych zawiera również kilkadziesiąt przykładów zdjęć hard-mining (zdjęcia roślin bez defektów) w celu określenia wpływu takich danych na uczenie modeli segmentacyjnych.

| Zdjęcie | Maska|
|---------|------|
| ![Image_14](https://github.com/user-attachments/assets/507d878c-113e-4185-a14c-affa9943c8a2) | ![Image_14-mask](https://github.com/user-attachments/assets/e902e1ba-2704-4af6-aac6-c1db9c03ad9b) |

```
Skopiuj repozytorium za pomocą `git clone`
```
```
Stwórz wirtualne środowisko `python -m venv venv`
```
```
Aktywuj wirtualne środkowisko `.\venv\Scripts\activate`
```
```
Pobierz biblioteki `pip install -r requirements.txt`
```
```
Pobierz dane i umieść je w folderze data
```
```
Następnie wpisz dvc pull
```
```
Aby uruchomić pipeline wpisz `dvc repro`
```
