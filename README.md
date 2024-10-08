##  Модерация карточек товаров с признаками нарушений правил площадки
19.08.2024 19:00 - 08.09.2024

Computer visionML

Участникам хакатона предстоить разработать ML-модель, которая будет определять на фото акт курения
и отмечать такие фотографии.
Модель должна найти среди представленных изображений как можно больше фото,
где располагается признаки нарушения правил площадки.

##  Запуск
##### Склонируйте репозиторий
```
git clone https://github.com/OlgaTora/E-CUP-Codenrock.git
```

##### загрузите обучающие данные
1. Скачайте train набор и поменяйте под него код в файле dataset.py
2. Разархивируйте содержимое архива в папку data таким образом, чтобы все изображения лежали в папке data/train

##### Соберите образ решения
```shell
docker build . -t moderatsiya
```
Это займёт какое-то время. Обратите внимание, что все пакеты, модели и другие данные, которые, которые вы хотите загрузить из интернета и использовать в проекте, должны быть загружены на этом этапе. В дальнейшем работа программы будет оторвана от интернета и ни скачать, ни выкачать ничего не сможет.

##### Запустите baseline.py
```shell
docker run -it --network none --shm-size 2G --name moderatsiya -v ./data:/app/data moderatsiya python baseline.py
```
##### Запустите инференс make submission py
```shell
docker run -it --network none --shm-size 2G --name moderatsiya -v ./data:/app/data moderatsiya python make_submission.py
```
