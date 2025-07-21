import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
from datetime import datetime

class CNNClassifier(nn.Module): #Сверточная нейронная сеть
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CNNClassifier, self).__init__()

        #Сверточные слои с пакетной нормализацией
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #Адаптивный пулинг для гибкого размера входа
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        #Полносвязные слои с dropout для неопределенности
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

        #Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self): #Инициализация Ксавье для лучшей сходимости
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        #Прямой проход
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        if x.device.type == 'mps':
            #Для MPS используем ручное изменение размера
            x = F.adaptive_avg_pool2d(x.cpu(), (4, 4)).to(x.device)
        else:
            x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)

        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

    def predict_with_uncertainty(self, x, num_samples=100): #Monte Carlo Dropout для оценки неопределенности

        self.train()
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                pred = F.softmax(self.forward(x), dim=1)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)

        return mean_pred, uncertainty


class UncertaintyDataset(Dataset): #Датасет для хранения изображений с неопределенностью
    def __init__(self, images, labels, uncertainties, transform=None):
        self.images = images
        self.labels = labels
        self.uncertainties = uncertainties
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        uncertainty = self.uncertainties[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, uncertainty


class HITLTrainingSystem: #Система обучения нейронной сети с участием человека
    def __init__(self, model, device, uncertainty_dir="uncertain_samples",
                 uncertainty_threshold=0.3, confidence_threshold=0.7):
        self.model = model
        self.device = device
        self.uncertainty_dir = uncertainty_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold

        #Создание директорий
        os.makedirs(uncertainty_dir, exist_ok=True)
        for i in range(10):  # Для цифр 0-9
            os.makedirs(f"{uncertainty_dir}/digit_{i}", exist_ok=True)

        #Метрики обучения
        self.training_history = {
            'initial_accuracy': 0,
            'current_accuracy': 0,
            'human_feedback_count': 0,
            'improvement_from_human': 0,
            'epoch_accuracies': [],
            'human_feedback_accuracies': []
        }

        #Логирование
        self.log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def train_initial_model(self, train_loader, val_loader, epochs=10): #Первоначальное обучение модели
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_accuracy = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')

            val_accuracy = self.evaluate_model(val_loader)
            self.training_history['epoch_accuracies'].append(val_accuracy)

            print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss / len(train_loader):.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')

            scheduler.step()

        self.training_history['initial_accuracy'] = best_accuracy
        self.training_history['current_accuracy'] = best_accuracy
        print(f"Начальное обучение завершено. Точность: {best_accuracy:.2f}%")

        return best_accuracy

    def evaluate_model(self, test_loader): #Оценка модели на тестовых данных
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def find_uncertain_samples(self, data_loader, num_samples=50): #Поиск неопределенных образцов с использованием Monte Carlo Dropout
        uncertain_samples = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Получение предсказаний с uncertainty
                mean_pred, uncertainty = self.model.predict_with_uncertainty(data)

                # Вычисление энтропии как меры неопределенности
                entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
                max_confidence = np.max(mean_pred, axis=1)

                for i in range(len(data)):
                    if (entropy[i] > self.uncertainty_threshold and
                            max_confidence[i] < self.confidence_threshold):

                        # Сохранение неопределенного образца
                        img_tensor = data[i].cpu()
                        true_label = target[i].item()
                        predictions = mean_pred[i]

                        # Получение топ-3 предсказаний
                        top_indices = np.argsort(predictions)[-3:][::-1]
                        top_predictions = [(idx, predictions[idx]) for idx in top_indices]

                        uncertain_samples.append({
                            'image': img_tensor,
                            'true_label': true_label,
                            'predictions': top_predictions,
                            'entropy': entropy[i],
                            'max_confidence': max_confidence[i]
                        })

                        if len(uncertain_samples) >= num_samples:
                            break

                if len(uncertain_samples) >= num_samples:
                    break

        print(f"Найдено {len(uncertain_samples)} неопределенных образцов")
        return uncertain_samples

    def save_uncertain_samples(self, uncertain_samples): #Сохранение неопределенных образцов в папки
        for idx, sample in enumerate(uncertain_samples):
            image = sample['image']
            true_label = sample['true_label']
            predictions = sample['predictions']

            #Преобразование тензора в изображение
            img_np = image.squeeze().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

            #Сохранение в папку наиболее вероятного класса
            predicted_label = predictions[0][0]
            filename = f"uncertain_{idx}_true_{true_label}_pred_{predicted_label}.png"
            filepath = os.path.join(self.uncertainty_dir, f"digit_{predicted_label}", filename)

            img_pil.save(filepath)

            #Сохранение метаданных с правильным преобразованием типов
            metadata = {
                'true_label': int(true_label),  #Преобразование в Python int
                'predictions': [[int(pred[0]), float(pred[1])] for pred in predictions],  #Преобразование типов
                'entropy': float(sample['entropy']),
                'max_confidence': float(sample['max_confidence'])
            }

            metadata_path = filepath.replace('.png', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def human_feedback_session(self, uncertain_samples): #Сессия получения обратной связи от человека
        corrected_samples = []
        print("Начало сессии обратной связи")

        self.model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(uncertain_samples):
                image = sample['image'].to(self.device).unsqueeze(0)

                #Получаем вероятности предсказания
                outputs = torch.stack([self.model(image) for _ in range(10)])
                probs = F.softmax(outputs, dim=2).mean(dim=0).squeeze()  #Средняя вероятность

                #Топ 3 предположения
                top_probs, top_indices = torch.topk(probs, k=3)

                # Вывод возможных варианты
                print("Возможные цифры модели:")
                for i in range(3):
                    print(f"{i + 1}. Цифра {top_indices[i].item()} — вероятность {top_probs[i].item():.2%}")

                #Отображение изображений
                plt.imshow(sample['image'].squeeze().cpu().numpy(), cmap='gray')
                plt.title(f"Неуверенное предсказание ({idx + 1}/{len(uncertain_samples)})")
                plt.axis('off')
                plt.show()

                # Получение метки от пользователя
                while True:
                    user_input = input("Введите правильную цифру (0–9), или 's' чтобы пропустить: ")
                    if user_input.lower() == 's':
                        break
                    try:
                        corrected_label = int(user_input)
                        if 0 <= corrected_label <= 9:
                            corrected_samples.append({
                                'image': sample['image'].cpu(),
                                'corrected_label': corrected_label
                            })
                            self.training_history['human_feedback_count'] += 1
                            break
                        else:
                            print("Цифра должна быть от 0 до 9")
                    except ValueError:
                        print("Неверный ввод. Введите число от 0 до 9 или 's'.")

        print(f"Получено корректировок: {len(corrected_samples)}")
        return corrected_samples

    def retrain_with_feedback(self, train_loader, corrected_samples, epochs=5): #Дообучение модели с использованием обратной связи
        print("Дообучение модели с обратной связью...")

        #Создание нового датасета с корректировками
        corrected_images = [sample['image'] for sample in corrected_samples]
        corrected_labels = [sample['corrected_label'] for sample in corrected_samples]

        #Преобразование в тензоры
        corrected_images_tensor = torch.stack(corrected_images)
        corrected_labels_tensor = torch.tensor(corrected_labels, dtype=torch.long)

        #Создание DataLoader для корректированных данных
        corrected_dataset = torch.utils.data.TensorDataset(
            corrected_images_tensor, corrected_labels_tensor
        )
        corrected_loader = DataLoader(corrected_dataset, batch_size=32, shuffle=True)

        #Дообучение
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  #Меньший learning rate

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0

            #Обучение на корректированных данных
            for data, target in corrected_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            #Обучение на части оригинальных данных (для предотвращения забывания)
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx > 10:  #Ограничиваем количество батчей
                    break

                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(corrected_loader):.4f}")

    def run_hitl_training(self, train_loader, val_loader, initial_epochs=10, hitl_iterations=3, uncertain_samples_per_iteration=20): #Полный цикл обучения с участием человека
        #Начальное обучение
        initial_accuracy = self.train_initial_model(train_loader, val_loader, initial_epochs)

        #Итерации с участием человека
        for iteration in range(hitl_iterations):
            print(f"\n Итерация HITL {iteration + 1}/{hitl_iterations}")
            print("-" * 40)

            #Поиск неопределенных образцов
            uncertain_samples = self.find_uncertain_samples(
                val_loader, uncertain_samples_per_iteration
            )

            if not uncertain_samples:
                break

            #Сохранение неопределенных образцов
            self.save_uncertain_samples(uncertain_samples)

            #Получение обратной связи от человека
            corrected_samples = self.human_feedback_session(uncertain_samples)

            #Дообучение модели
            self.retrain_with_feedback(train_loader, corrected_samples)

            #Оценка улучшения
            new_accuracy = self.evaluate_model(val_loader)
            improvement = new_accuracy - self.training_history['current_accuracy']

            self.training_history['current_accuracy'] = new_accuracy
            self.training_history['improvement_from_human'] += improvement
            self.training_history['human_feedback_accuracies'].append(new_accuracy)

            print(f"Точность после итерации {iteration + 1}: {new_accuracy:.2f}%")
            print(f" Улучшение: +{improvement:.2f}%")

        #Финальная статистика
        self.print_final_statistics()
        self.save_training_log()

        return self.training_history

    def print_final_statistics(self): #Вывод финальной статистики обучения
        initial_acc = self.training_history['initial_accuracy']
        current_acc = self.training_history['current_accuracy']
        human_improvement = self.training_history['improvement_from_human']
        feedback_count = self.training_history['human_feedback_count']

        print(f"Начальная точность: {initial_acc:.2f}%")
        print(f"Финальная точность: {current_acc:.2f}%")
        print(f"Улучшение благодаря человеку: +{human_improvement:.2f}%")
        print(f"Количество корректировок: {feedback_count}")

        if initial_acc > 0:
            relative_improvement = ((current_acc - initial_acc) / initial_acc) * 100
            print(f"Относительное улучшение: +{relative_improvement:.1f}%")

    def save_training_log(self): #Сохранение лога обучения
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'model_parameters': {
                'uncertainty_threshold': self.uncertainty_threshold,
                'confidence_threshold': self.confidence_threshold
            }
        }

        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def visualize_training_progress(self): #Визуализация прогресса обучения
        plt.figure(figsize=(12, 8))

        #График точности по эпохам
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history['epoch_accuracies'], 'b-', label='Обычное обучение')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность (%)')
        plt.title('Прогресс обучения')
        plt.legend()
        plt.grid(True)

        #График улучшений от человека
        plt.subplot(2, 2, 2)
        hitl_accuracies = self.training_history['human_feedback_accuracies']
        if hitl_accuracies:
            plt.plot(hitl_accuracies, 'r-o', label='С участием человека')
            plt.xlabel('Итерация HITL')
            plt.ylabel('Точность (%)')
            plt.title('Улучшение с участием человека')
            plt.legend()
            plt.grid(True)

        #Сравнение методов
        plt.subplot(2, 2, 3)
        methods = ['Начальная', 'Финальная']
        accuracies = [
            self.training_history['initial_accuracy'],
            self.training_history['current_accuracy']
        ]
        plt.bar(methods, accuracies, color=['skyblue', 'lightgreen'])
        plt.ylabel('Точность (%)')
        plt.title('Сравнение точности')
        plt.ylim(0, 100)

        #Добавление значений на столбцы
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

def load_custom_dataset(train_dir, test_dir, image_size=(28, 28)):
    def load_custom_dataset(train_dir, test_dir, image_size=(28, 28)):  # Загрузка датасетов
        # Проверка существования директорий
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Директория {train_dir} не найдена")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Директория {test_dir} не найдена")

    class AddGaussianNoise: #Гауссовский шум
        def __init__(self, mean=0.0, std=0.1):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    # Определение transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.13),  #Можно изменить шум
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #Загрузка датасетов
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    return train_dataset, test_dataset

def main(): #Основная функция для запуска системы обучения

    #Настройка устройства (с приоритетом MPS для Mac)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Используемое устройство: MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Используемое устройство: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Используемое устройство: CPU")

    #Пути к датасетам
    train_dir = "Trainingimages"
    test_dir = "Testimages"

    #Загрузка датасетов
    try:
        train_dataset, test_dataset = load_custom_dataset(train_dir, test_dir)
        print("Датасеты успешно загружены")
    except Exception as e:
        print(f"Ошибка при загрузке датасетов: {e}")

    # Создание DataLoader'ов (для MPS лучше меньше workers)
    num_workers = 0 if device.type == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)

    # Создание модели
    model = CNNClassifier(num_classes=10).to(device)

    # Создание системы обучения с участием человека
    hitl_system = HITLTrainingSystem(
        model=model,
        device=device,
        uncertainty_threshold=0.5,
        confidence_threshold=0.8
    )

    # Запуск обучения
    training_history = hitl_system.run_hitl_training(
        train_loader=train_loader,
        val_loader=test_loader,
        initial_epochs=3,
        hitl_iterations=2,
        uncertain_samples_per_iteration=20
    )

    # Визуализация результатов
    hitl_system.visualize_training_progress()

    return training_history


if __name__ == "__main__":
    # Установка seed для воспроизводимости
    torch.manual_seed(42)
    np.random.seed(42)

    # Запуск системы
    history = main()

    print("Обучение полностью завершено")

