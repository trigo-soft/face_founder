import cv2
import urllib.request

# Загрузка классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка модели для определения пола
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# Загрузка списка меток для пола
gender_list = ['Male', 'Female']

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра с веб-камеры
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Обработка каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Отрисовка зеленого квадрата вокруг лица
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Извлечение области лица
        face_roi = frame[y:y+h, x:x+w]

        # Подготовка области лица для определения пола
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Передача области лица через модель для определения пола
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()

        # Получение индекса класса с наибольшей вероятностью
        gender_idx = gender_preds[0].argmax()
        gender = gender_list[gender_idx]

        # Отображение пола под квадратом
        label = f'Gender: {gender}'
        cv2.putText(frame, label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Отображение кадра с обведенными лицами и полом
    cv2.imshow('Face Detection', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
