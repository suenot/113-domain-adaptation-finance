# Глава 92: Доменная адаптация для финансов

## Обзор

**Доменная адаптация (Domain Adaptation)** -- раздел трансферного обучения, который решает задачу переноса модели, обученной на одном распределении данных (исходный домен), в новое распределение (целевой домен), где размеченные данные ограничены или полностью отсутствуют. В финансах это критическая проблема: модель, обученная на исторических данных рынка акций США, может плохо работать на криптовалютном рынке, на бирже другой страны или даже на тех же данных в изменившемся рыночном режиме. Доменный сдвиг (domain shift) -- фундаментальная причина деградации торговых стратегий.

В этой главе мы рассмотрим теорию и практику доменной адаптации применительно к алгоритмическому трейдингу. Мы начнём с математических основ -- теоремы Бен-Дэвида, максимального среднего расхождения (MMD), доменно-состязательных сетей (DANN) и метода корреляционного выравнивания (CORAL). Затем реализуем каждый из этих методов на Python (PyTorch) и Rust, продемонстрируем их работу на реальных данных акций и криптовалют (Bybit), и построим систему бэктестинга для оценки эффективности адаптированных моделей.

## Содержание

1. [Введение в доменную адаптацию](#введение-в-доменную-адаптацию)
2. [Математические основы](#математические-основы)
3. [Доменная адаптация vs другие методы переноса](#доменная-адаптация-vs-другие-методы-переноса)
4. [Применение в трейдинге](#применение-в-трейдинге)
5. [Реализация на Python](#реализация-на-python)
6. [Реализация на Rust](#реализация-на-rust)
7. [Практические примеры с данными акций и криптовалют](#практические-примеры-с-данными-акций-и-криптовалют)
8. [Фреймворк бэктестинга](#фреймворк-бэктестинга)
9. [Оценка производительности](#оценка-производительности)
10. [Перспективные направления](#перспективные-направления)

---

## Введение в доменную адаптацию

### Что такое доменная адаптация?

Доменная адаптация решает задачу, возникающую когда распределение обучающих данных (исходный домен $\mathcal{D}_S$) отличается от распределения данных, на которых модель будет использоваться (целевой домен $\mathcal{D}_T$). Формально:

- **Исходный домен**: $\mathcal{D}_S = \{(x_i^s, y_i^s)\}_{i=1}^{n_s}$ -- размеченные данные из исходного распределения $P_S(X, Y)$
- **Целевой домен**: $\mathcal{D}_T = \{x_j^t\}_{j=1}^{n_t}$ -- данные из целевого распределения $P_T(X, Y)$, обычно без меток
- **Цель**: обучить модель $f: X \rightarrow Y$, минимизирующую ошибку на целевом домене $\epsilon_T(f)$

Типы доменной адаптации:

| Тип | Целевые метки | Описание |
|-----|---------------|----------|
| **Unsupervised DA** | Нет | Наиболее распространённый: метки только в исходном домене |
| **Semi-supervised DA** | Немного | Небольшое количество меток в целевом домене |
| **Supervised DA** | Есть | Метки в обоих доменах, но с разными распределениями |

### Проблема доменного сдвига в финансах

Финансовые данные подвержены доменному сдвигу по нескольким причинам:

**Временной сдвиг (Temporal Shift)**:
- Рыночные режимы меняются: бычий рынок 2020-2021 vs медвежий рынок 2022
- Волатильность не стационарна: VIX может колебаться от 10 до 80
- Макроэкономические условия эволюционируют: нулевые ставки vs повышение ставок

**Кросс-рыночный сдвиг (Cross-Market Shift)**:
- Акции vs криптовалюты: разная микроструктура, часы торгов, ликвидность
- Развитые vs развивающиеся рынки: разная регуляция, состав участников
- Разные классы активов: облигации vs товары vs FX

**Кросс-биржевой сдвиг (Cross-Exchange Shift)**:
- NYSE vs NASDAQ: разные правила маркет-мейкинга
- Binance vs Bybit: разная ликвидность, комиссии, типы ордеров
- Централизованные vs децентрализованные биржи

```
Пример доменного сдвига:

Модель обучена на:                Применяется к:
┌─────────────────────┐          ┌─────────────────────┐
│ S&P 500 акции       │          │ Криптовалюты Bybit   │
│ 2018-2022           │  ──DA──> │ 2024-2025            │
│ Дневные данные      │          │ Часовые данные       │
│ Низкая волатильность│          │ Высокая волатильность│
└─────────────────────┘          └─────────────────────┘
  P_S(X,Y) ≠ P_T(X,Y)
```

### Почему доменная адаптация для трейдинга?

Классические подходы к проблеме доменного сдвига в трейдинге -- переобучение модели на новых данных или fine-tuning -- имеют существенные ограничения:

1. **Недостаток размеченных данных**: в целевом домене может быть слишком мало данных для обучения с нуля
2. **Временные и вычислительные затраты**: полное переобучение дорого и медленно
3. **Катастрофическое забывание**: fine-tuning может разрушить полезные знания из исходного домена
4. **Задержка адаптации**: пока накопятся данные для переобучения, рыночный режим может снова измениться

Доменная адаптация решает эти проблемы, обучая **доменно-инвариантные представления** -- признаки, которые полезны для предсказания вне зависимости от домена. Это позволяет:

- Переносить модели между рынками без переобучения
- Адаптироваться к новым режимам с минимумом данных
- Сохранять знания из исходного домена
- Ускорять развертывание на новых биржах и классах активов

---

## Математические основы

### Теория доменной адаптации

Формально определим ключевые понятия:

**Домен** -- пара $(\mathcal{X}, P(X))$, где $\mathcal{X}$ -- пространство признаков, $P(X)$ -- маргинальное распределение.

**Задача** -- пара $(\mathcal{Y}, P(Y|X))$, где $\mathcal{Y}$ -- пространство меток, $P(Y|X)$ -- условное распределение.

**Доменный сдвиг** возникает когда:
- **Covariate shift**: $P_S(X) \neq P_T(X)$, но $P_S(Y|X) = P_T(Y|X)$ -- наиболее изученный случай
- **Label shift**: $P_S(Y) \neq P_T(Y)$, но $P_S(X|Y) = P_T(X|Y)$
- **Concept shift**: $P_S(Y|X) \neq P_T(Y|X)$ -- наиболее сложный случай
- **Full shift**: $P_S(X,Y) \neq P_T(X,Y)$ -- всё отличается

#### Граница Бен-Дэвида для ошибки на целевом домене

Фундаментальная теорема Бен-Дэвида (Ben-David et al., 2010) устанавливает верхнюю границу ошибки на целевом домене:

$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2} d_{\mathcal{H} \Delta \mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda^*$$

Где:
- $\epsilon_T(h)$ -- ошибка гипотезы $h$ на целевом домене
- $\epsilon_S(h)$ -- ошибка на исходном домене (минимизируется обучением)
- $d_{\mathcal{H} \Delta \mathcal{H}}$ -- $\mathcal{H}$-расхождение между доменами (минимизируется адаптацией)
- $\lambda^* = \min_{h \in \mathcal{H}} [\epsilon_S(h) + \epsilon_T(h)]$ -- ошибка идеальной совместной гипотезы (константа)

Эта теорема мотивирует два ключевых принципа:
1. **Минимизировать ошибку на исходном домене** -- стандартное обучение
2. **Минимизировать расхождение между доменами** -- доменная адаптация

### Максимальное среднее расхождение (MMD)

**Maximum Mean Discrepancy (MMD)** -- ядерный метод оценки расстояния между распределениями. MMD определяется как:

$$\text{MMD}^2(\mathcal{D}_S, \mathcal{D}_T) = \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} \phi(x_i^s) - \frac{1}{n_t} \sum_{j=1}^{n_t} \phi(x_j^t) \right\|_{\mathcal{H}_k}^2$$

Где $\phi: \mathcal{X} \rightarrow \mathcal{H}_k$ -- отображение в воспроизводящее ядерное гильбертово пространство (RKHS), ассоциированное с ядром $k$.

С использованием ядерного трюка:

$$\text{MMD}^2 = \frac{1}{n_s^2} \sum_{i,i'} k(x_i^s, x_{i'}^s) - \frac{2}{n_s n_t} \sum_{i,j} k(x_i^s, x_j^t) + \frac{1}{n_t^2} \sum_{j,j'} k(x_j^t, x_{j'}^t)$$

Обычно используется гауссовское RBF-ядро:

$$k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

**Для доменной адаптации** MMD добавляется как регуляризатор к функции потерь:

$$\mathcal{L} = \mathcal{L}_{task}(\hat{y}_S, y_S) + \lambda \cdot \text{MMD}^2(f(X_S), f(X_T))$$

Где $f$ -- feature extractor, $\lambda$ -- вес регуляризации.

### Доменно-состязательные нейронные сети (DANN)

**Domain-Adversarial Neural Network (DANN)**, предложенная Ganin et al. (2016), использует состязательное обучение для выравнивания доменов. Архитектура состоит из трёх компонентов:

```
Вход x
  │
  ├── Feature Extractor G_f(x; θ_f) ──→ Признаки z
  │                                         │
  │                        ┌────────────────┤
  │                        │                │
  │                        ▼                ▼
  │              Label Predictor     Domain Classifier
  │              G_y(z; θ_y)         G_d(z; θ_d)
  │                   │                    │
  │                   ▼                    ▼
  │              Предсказание ŷ      Домен d̂ ∈ {S, T}
  │
  └── Gradient Reversal Layer (GRL): ∇ → -λ∇
```

**Gradient Reversal Layer (GRL)** -- ключевая инновация DANN:
- При прямом проходе: тождественная функция $\text{GRL}(x) = x$
- При обратном проходе: инвертирует градиент $\frac{\partial \text{GRL}}{\partial x} = -\lambda I$

Это создаёт **минимаксную игру**:

$$\mathcal{L}(\theta_f, \theta_y, \theta_d) = \mathcal{L}_y(\theta_f, \theta_y) - \lambda \mathcal{L}_d(\theta_f, \theta_d)$$

$$\hat{\theta}_f, \hat{\theta}_y = \arg\min_{\theta_f, \theta_y} \mathcal{L}$$
$$\hat{\theta}_d = \arg\max_{\theta_d} \mathcal{L}$$

Feature extractor учится создавать признаки, которые:
1. Полезны для предсказания меток (минимизация $\mathcal{L}_y$)
2. Не позволяют отличить домены (максимизация $\mathcal{L}_d$ через GRL)

### Корреляционное выравнивание (CORAL)

**CORrelation ALignment (CORAL)**, предложенный Sun et al. (2016), выравнивает статистики второго порядка (ковариационные матрицы) исходного и целевого доменов.

Для признаков $D_S \in \mathbb{R}^{n_s \times d}$ и $D_T \in \mathbb{R}^{n_t \times d}$:

$$\mathcal{L}_{CORAL} = \frac{1}{4d^2} \|C_S - C_T\|_F^2$$

Где:
- $C_S = \frac{1}{n_s - 1}(D_S^\top D_S - \frac{1}{n_s}(\mathbf{1}^\top D_S)^\top (\mathbf{1}^\top D_S))$ -- ковариационная матрица исходного домена
- $C_T$ -- аналогично для целевого домена
- $\|\cdot\|_F$ -- норма Фробениуса

**Deep CORAL** интегрирует CORAL в глубокие сети:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{CORAL}(f_S, f_T)$$

Где $f_S$ и $f_T$ -- выходы feature extractor для исходных и целевых данных соответственно.

Преимущества CORAL:
- Простота реализации
- Дифференцируемость (можно обучать end-to-end)
- Не требует выбора ядра (в отличие от MMD)
- Вычислительная эффективность: $O(d^2)$ по размерности признаков

### Оптимальный транспорт для доменной адаптации

**Оптимальный транспорт (Optimal Transport)** ищет наиболее экономичный способ преобразовать одно распределение в другое. Расстояние Вассерштейна:

$$W_p(\mu, \nu) = \left(\inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} \|x - y\|^p \, d\gamma(x, y)\right)^{1/p}$$

Где $\Gamma(\mu, \nu)$ -- множество всех совместных распределений с маргиналами $\mu$ и $\nu$.

Для дискретных распределений (samples) задача сводится к линейной программе:

$$W_p = \min_{T \in \Pi(a, b)} \sum_{i,j} T_{ij} \cdot c(x_i^s, x_j^t)^p$$

Где:
- $T$ -- план транспортировки
- $\Pi(a, b)$ -- множество допустимых планов
- $c(x, y)$ -- стоимость перемещения

**Регуляризация Синкхорна** делает вычисления эффективными:

$$W_\varepsilon = \min_{T \in \Pi(a, b)} \sum_{i,j} T_{ij} c_{ij} + \varepsilon \sum_{i,j} T_{ij} \log T_{ij}$$

Применение в финансах:
- Выравнивание распределений доходностей между рынками
- Поиск соответствий между активами разных бирж
- Отслеживание эволюции распределения портфеля

---

## Доменная адаптация vs другие методы переноса

| Характеристика | Fine-tuning | Доменная адаптация | Multi-task Learning | Meta-Learning |
|----------------|-------------|-------------------|---------------------|---------------|
| **Целевые метки** | Необходимы | Не обязательны | Необходимы для всех задач | Немного |
| **Архитектура** | Та же | Расширенная (GRL, MMD loss) | Общий backbone + heads | Модель моделей |
| **Знания исходного домена** | Частично теряются | Сохраняются | Усиливаются | Обобщаются |
| **Вычислительная стоимость** | Низкая | Средняя | Высокая | Высокая |
| **Скорость адаптации** | Средняя | Высокая | Низкая | Очень высокая |
| **Риск переобучения** | Высокий (на малых данных) | Низкий | Средний | Низкий |
| **Применимость к трейдингу** | Базовый перенос | Кросс-рыночная адаптация | Мульти-актив портфели | Few-shot стратегии |
| **Пример** | Дообучение GPT на финансовых текстах | Акции $\rightarrow$ крипто | Одновременно акции + облигации + FX | Быстрая адаптация к новому активу |

---

## Применение в трейдинге

### 1. Кросс-рыночная адаптация (акции -> криптовалюты)

Модель, обученная на данных фондового рынка, адаптируется для работы на криптовалютном рынке.

```
Исходный домен: S&P 500          Целевой домен: Bybit Top-20
├── Дневные OHLCV                 ├── Часовые OHLCV
├── 20+ лет истории               ├── 5-7 лет истории
├── Регулируемый рынок            ├── Менее регулируемый
├── Торги 6.5ч/день               ├── Торги 24/7
└── Волатильность 15-20% годовых  └── Волатильность 60-100% годовых
```

**Доменный сдвиг**: распределения доходностей, волатильности и объёмов значительно отличаются. Однако паттерны моментума, mean-reversion и микроструктурные сигналы имеют общую природу.

**Подход**: DANN с feature extractor, обученным на обоих рынках одновременно. Gradient Reversal Layer заставляет модель находить общие паттерны, абстрагируясь от различий рынков.

### 2. Временная доменная адаптация (смена режимов)

Адаптация модели при переходе между рыночными режимами без полного переобучения.

```
Режим 1 (бычий):                 Режим 2 (медвежий):
├── Восходящий тренд              ├── Нисходящий тренд
├── Низкая волатильность          ├── Высокая волатильность
├── Высокая корреляция активов    ├── Ещё более высокая корреляция
└── Momentum работает             └── Mean-reversion работает
```

**Подход**: MMD-регуляризация для минимизации расхождения между скользящими окнами данных. Модель обучается на историческом окне (исходный домен) и адаптируется к текущему окну (целевой домен), обновляя выравнивание в реальном времени.

### 3. Кросс-биржевая адаптация

Перенос модели, обученной на данных одной биржи, на другую.

```
Bybit (исходный):                Binance (целевой):
├── Своя книга ордеров            ├── Другая книга ордеров
├── Своя комиссионная структура   ├── Другая комиссионная структура
├── Своя ликвидность              ├── Другая ликвидность
└── Своя задержка API             └── Другая задержка API
```

**Подход**: CORAL для выравнивания статистик ордербука между биржами. Ковариационные матрицы признаков (spread, depth, imbalance) выравниваются между биржами.

### 4. Кросс-активная адаптация

Перенос знаний между классами активов: облигации, товары, валюты.

| Перенос | Общие паттерны | Различия |
|---------|---------------|----------|
| Акции $\rightarrow$ FX | Momentum, mean-reversion | Leverage, carry trade |
| Акции $\rightarrow$ Commodities | Технические паттерны | Сезонность, contango |
| FX $\rightarrow$ Крипто | Круглосуточная торговля | Волатильность, регуляция |
| Облигации $\rightarrow$ Акции | Макро-факторы | Кредитный риск |

**Подход**: Оптимальный транспорт для нахождения соответствий между распределениями активов разных классов. Wasserstein distance служит мерой «расстояния» между активами.

---

## Реализация на Python

### DANN (Domain-Adversarial Neural Network)

Полная реализация DANN на PyTorch с Gradient Reversal Layer:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import numpy as np


class GradientReversalFunction(Function):
    """Gradient Reversal Layer (GRL) implementation.

    Forward pass: identity function.
    Backward pass: reverses gradient by multiplying with -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class FeatureExtractor(nn.Module):
    """Shared feature extractor for both domains."""

    def __init__(self, input_dim, hidden_dim=128, feature_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class LabelPredictor(nn.Module):
    """Predicts task labels (e.g., buy/sell/hold)."""

    def __init__(self, feature_dim=64, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, features):
        return self.network(features)


class DomainClassifier(nn.Module):
    """Classifies whether features come from source or target domain."""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.network(features)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network for financial domain adaptation.

    Architecture:
        Input -> FeatureExtractor -> [LabelPredictor, GRL -> DomainClassifier]

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension in feature extractor
        feature_dim: Output dimension of feature extractor
        num_classes: Number of prediction classes (e.g., 3 for buy/sell/hold)
        lambda_: GRL scaling factor (increases during training)
    """

    def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                 num_classes=3, lambda_=1.0):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim)
        self.label_predictor = LabelPredictor(feature_dim, num_classes)
        self.domain_classifier = DomainClassifier(feature_dim)
        self.grl = GradientReversalLayer(lambda_)

    def forward(self, x, alpha=1.0):
        self.grl.lambda_ = alpha
        features = self.feature_extractor(x)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output, features

    def predict(self, x):
        """Inference: only returns class predictions."""
        features = self.feature_extractor(x)
        return self.label_predictor(features)


def train_dann(model, source_loader, target_loader,
               num_epochs=100, lr=1e-3, device='cpu'):
    """Training loop for DANN with progressive lambda scheduling.

    Lambda increases from 0 to 1 during training (Ganin et al. schedule):
        lambda_p = 2 / (1 + exp(-10 * p)) - 1
    where p = epoch / num_epochs
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()

    model.to(device)
    history = {'task_loss': [], 'domain_loss': [], 'total_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_domain_loss = 0.0

        # Progressive lambda schedule (Ganin et al.)
        p = epoch / num_epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

        target_iter = iter(target_loader)

        for source_batch in source_loader:
            source_x, source_y = source_batch
            source_x = source_x.to(device)
            source_y = source_y.to(device)

            try:
                target_x = next(target_iter)[0].to(device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x = next(target_iter)[0].to(device)

            # Forward pass - source domain
            class_output, domain_output_s, _ = model(source_x, alpha)
            task_loss = task_criterion(class_output, source_y)
            domain_loss_s = domain_criterion(
                domain_output_s,
                torch.zeros(source_x.size(0), 1).to(device)  # source=0
            )

            # Forward pass - target domain
            _, domain_output_t, _ = model(target_x, alpha)
            domain_loss_t = domain_criterion(
                domain_output_t,
                torch.ones(target_x.size(0), 1).to(device)   # target=1
            )

            domain_loss = domain_loss_s + domain_loss_t
            total_loss = task_loss + domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_domain_loss += domain_loss.item()

        n_batches = len(source_loader)
        history['task_loss'].append(epoch_task_loss / n_batches)
        history['domain_loss'].append(epoch_domain_loss / n_batches)
        history['total_loss'].append(
            (epoch_task_loss + epoch_domain_loss) / n_batches
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Task Loss: {history['task_loss'][-1]:.4f} "
                  f"Domain Loss: {history['domain_loss'][-1]:.4f} "
                  f"Alpha: {alpha:.4f}")

    return history
```

### MMD (Maximum Mean Discrepancy)

Реализация MMD с гауссовским ядром и мультиядерным вариантом:

```python
import torch
import torch.nn as nn


def gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian RBF kernel matrix between x and y.

    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)

    kernel = torch.exp(-torch.sum((x - y) ** 2, dim=2) / (2 * sigma ** 2))
    return kernel


def compute_mmd(source, target, kernel_bandwidths=[0.1, 0.5, 1.0, 5.0, 10.0]):
    """Compute MMD^2 with multiple Gaussian kernels (multi-kernel MMD).

    MMD^2 = E[k(xs, xs')] - 2*E[k(xs, xt)] + E[k(xt, xt')]

    Using multiple bandwidths makes the statistic more robust.
    """
    mmd = 0.0
    for sigma in kernel_bandwidths:
        k_ss = gaussian_kernel(source, source, sigma)
        k_tt = gaussian_kernel(target, target, sigma)
        k_st = gaussian_kernel(source, target, sigma)

        mmd += k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

    return mmd / len(kernel_bandwidths)


class MMDAdaptationNetwork(nn.Module):
    """Neural network with MMD regularization for domain adaptation.

    Adds MMD penalty to align feature distributions between domains.
    """

    def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                 num_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = self.classifier(features)
        return predictions, features

    def extract_features(self, x):
        return self.feature_extractor(x)


def train_mmd_network(model, source_loader, target_loader,
                      num_epochs=100, lr=1e-3, mmd_weight=1.0,
                      device='cpu'):
    """Train network with MMD regularization.

    Loss = CrossEntropy(source) + mmd_weight * MMD^2(features_s, features_t)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss()

    model.to(device)
    history = {'task_loss': [], 'mmd_loss': [], 'total_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_mmd_loss = 0.0
        target_iter = iter(target_loader)

        for source_x, source_y in source_loader:
            source_x = source_x.to(device)
            source_y = source_y.to(device)

            try:
                target_x = next(target_iter)[0].to(device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x = next(target_iter)[0].to(device)

            # Forward pass
            source_pred, source_features = model(source_x)
            _, target_features = model(target_x)

            # Task loss (source only)
            task_loss = task_criterion(source_pred, source_y)

            # MMD loss (align features)
            mmd_loss = compute_mmd(source_features, target_features)

            # Combined loss
            total_loss = task_loss + mmd_weight * mmd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_mmd_loss += mmd_loss.item()

        n_batches = len(source_loader)
        history['task_loss'].append(epoch_task_loss / n_batches)
        history['mmd_loss'].append(epoch_mmd_loss / n_batches)
        history['total_loss'].append(
            (epoch_task_loss + mmd_weight * epoch_mmd_loss) / n_batches
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Task: {history['task_loss'][-1]:.4f} "
                  f"MMD: {history['mmd_loss'][-1]:.6f}")

    return history
```

### CORAL (CORrelation ALignment)

Реализация Deep CORAL:

```python
import torch
import torch.nn as nn


def coral_loss(source_features, target_features):
    """Compute CORAL loss: Frobenius norm of covariance difference.

    L_CORAL = (1 / 4d^2) * ||C_S - C_T||_F^2

    where C_S, C_T are covariance matrices of source and target features.
    """
    d = source_features.size(1)
    ns = source_features.size(0)
    nt = target_features.size(0)

    # Center the features
    source_centered = source_features - source_features.mean(0, keepdim=True)
    target_centered = target_features - target_features.mean(0, keepdim=True)

    # Compute covariance matrices
    cov_source = (source_centered.t() @ source_centered) / (ns - 1)
    cov_target = (target_centered.t() @ target_centered) / (nt - 1)

    # CORAL loss: squared Frobenius norm of difference
    loss = (cov_source - cov_target).pow(2).sum() / (4 * d * d)
    return loss


class DeepCORAL(nn.Module):
    """Deep CORAL network for domain adaptation.

    Minimizes classification loss + CORAL loss on last hidden layer.
    """

    def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                 num_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = self.classifier(features)
        return predictions, features


def train_deep_coral(model, source_loader, target_loader,
                     num_epochs=100, lr=1e-3, coral_weight=1.0,
                     device='cpu'):
    """Train Deep CORAL model.

    Loss = CrossEntropy(source) + coral_weight * CORAL(features_s, features_t)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss()

    model.to(device)
    history = {'task_loss': [], 'coral_loss': [], 'total_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_coral_loss = 0.0
        target_iter = iter(target_loader)

        for source_x, source_y in source_loader:
            source_x = source_x.to(device)
            source_y = source_y.to(device)

            try:
                target_x = next(target_iter)[0].to(device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x = next(target_iter)[0].to(device)

            # Forward
            source_pred, source_features = model(source_x)
            _, target_features = model(target_x)

            task_loss = task_criterion(source_pred, source_y)
            c_loss = coral_loss(source_features, target_features)

            total_loss = task_loss + coral_weight * c_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_coral_loss += c_loss.item()

        n_batches = len(source_loader)
        history['task_loss'].append(epoch_task_loss / n_batches)
        history['coral_loss'].append(epoch_coral_loss / n_batches)
        history['total_loss'].append(
            (epoch_task_loss + coral_weight * epoch_coral_loss) / n_batches
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Task: {history['task_loss'][-1]:.4f} "
                  f"CORAL: {history['coral_loss'][-1]:.6f}")

    return history
```

### Оптимальный транспорт для доменной адаптации

```python
import torch
import torch.nn as nn
import numpy as np


def sinkhorn_distance(x, y, epsilon=0.1, n_iter=50):
    """Compute Sinkhorn approximation to Wasserstein distance.

    Uses entropic regularization for efficient computation.

    Args:
        x: Source samples (n_s, d)
        y: Target samples (n_t, d)
        epsilon: Regularization strength
        n_iter: Number of Sinkhorn iterations

    Returns:
        Approximate Wasserstein distance
    """
    n_s = x.size(0)
    n_t = y.size(0)

    # Cost matrix (squared Euclidean distances)
    C = torch.cdist(x, y, p=2).pow(2)

    # Uniform marginals
    mu = torch.ones(n_s, device=x.device) / n_s
    nu = torch.ones(n_t, device=x.device) / n_t

    # Sinkhorn iterations
    K = torch.exp(-C / epsilon)
    u = torch.ones(n_s, device=x.device)

    for _ in range(n_iter):
        v = nu / (K.t() @ u + 1e-8)
        u = mu / (K @ v + 1e-8)

    # Transport plan
    T = torch.diag(u) @ K @ torch.diag(v)

    # Wasserstein distance
    distance = (T * C).sum()
    return distance


class OTDomainAdapter(nn.Module):
    """Optimal Transport-based domain adaptation network.

    Uses Sinkhorn distance as alignment loss.
    """

    def __init__(self, input_dim, hidden_dim=128, feature_dim=64,
                 num_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = self.classifier(features)
        return predictions, features


def train_ot_adapter(model, source_loader, target_loader,
                     num_epochs=100, lr=1e-3, ot_weight=0.5,
                     epsilon=0.1, device='cpu'):
    """Train OT-based domain adaptation network.

    Loss = CrossEntropy(source) + ot_weight * W_epsilon(features_s, features_t)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_criterion = nn.CrossEntropyLoss()

    model.to(device)
    history = {'task_loss': [], 'ot_loss': [], 'total_loss': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_task_loss = 0.0
        epoch_ot_loss = 0.0
        target_iter = iter(target_loader)

        for source_x, source_y in source_loader:
            source_x = source_x.to(device)
            source_y = source_y.to(device)

            try:
                target_x = next(target_iter)[0].to(device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_x = next(target_iter)[0].to(device)

            source_pred, source_features = model(source_x)
            _, target_features = model(target_x)

            task_loss = task_criterion(source_pred, source_y)
            ot_loss = sinkhorn_distance(
                source_features, target_features, epsilon=epsilon
            )

            total_loss = task_loss + ot_weight * ot_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_ot_loss += ot_loss.item()

        n_batches = len(source_loader)
        history['task_loss'].append(epoch_task_loss / n_batches)
        history['ot_loss'].append(epoch_ot_loss / n_batches)
        history['total_loss'].append(
            (epoch_task_loss + ot_weight * epoch_ot_loss) / n_batches
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Task: {history['task_loss'][-1]:.4f} "
                  f"OT: {history['ot_loss'][-1]:.4f}")

    return history
```

### Утилиты: подготовка финансовых данных

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class FinancialDomainDataset(Dataset):
    """Dataset for domain adaptation with financial data.

    Computes standard technical features from OHLCV data.
    """

    def __init__(self, df, label_col='label', window=20):
        self.features = self._compute_features(df, window)
        self.labels = df[label_col].values if label_col in df.columns else None

        # Standardize
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        self.features = torch.FloatTensor(self.features)

        if self.labels is not None:
            self.labels = torch.LongTensor(self.labels)

    def _compute_features(self, df, window):
        """Compute technical features from OHLCV data."""
        features = pd.DataFrame(index=df.index)

        # Returns at multiple horizons
        for h in [1, 5, 10, 20]:
            features[f'return_{h}d'] = df['close'].pct_change(h)

        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(window).std()
        features['volatility_ratio'] = (
            df['close'].pct_change().rolling(window // 2).std() /
            df['close'].pct_change().rolling(window).std()
        )

        # Volume features
        features['volume_sma_ratio'] = (
            df['volume'] / df['volume'].rolling(window).mean()
        )
        features['volume_std'] = df['volume'].rolling(window).std()

        # Price-based
        features['sma_ratio'] = (
            df['close'] / df['close'].rolling(window).mean()
        )
        features['high_low_range'] = (
            (df['high'] - df['low']) / df['close']
        )
        features['close_open_range'] = (
            (df['close'] - df['open']) / df['open']
        )

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close']

        features = features.dropna()
        return features.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx],


def create_trading_labels(df, forward_window=5, threshold=0.01):
    """Create trading labels: 0=sell, 1=hold, 2=buy.

    Based on forward returns exceeding threshold.
    """
    forward_returns = df['close'].pct_change(forward_window).shift(-forward_window)
    labels = pd.Series(1, index=df.index)  # default: hold
    labels[forward_returns > threshold] = 2   # buy
    labels[forward_returns < -threshold] = 0  # sell
    df['label'] = labels
    return df.dropna()
```

---

## Реализация на Rust

Оптимизированные реализации ключевых компонентов доменной адаптации на Rust для production-среды.

### MMD на Rust

```rust
use ndarray::{Array1, Array2, Axis};

/// Compute Gaussian RBF kernel between two vectors.
fn gaussian_kernel_single(x: &Array1<f64>, y: &Array1<f64>, sigma: f64) -> f64 {
    let diff = x - y;
    let sq_dist: f64 = diff.iter().map(|&v| v * v).sum();
    (-sq_dist / (2.0 * sigma * sigma)).exp()
}

/// Compute MMD^2 between source and target feature matrices.
///
/// Uses multi-kernel approach with multiple bandwidths.
///
/// # Arguments
/// * `source` - Source features (n_s x d)
/// * `target` - Target features (n_t x d)
/// * `bandwidths` - Kernel bandwidth parameters
///
/// # Returns
/// MMD^2 value
pub fn compute_mmd_squared(
    source: &Array2<f64>,
    target: &Array2<f64>,
    bandwidths: &[f64],
) -> f64 {
    let n_s = source.nrows();
    let n_t = target.nrows();
    let mut total_mmd = 0.0;

    for &sigma in bandwidths {
        // E[k(xs, xs')]
        let mut k_ss = 0.0;
        for i in 0..n_s {
            for j in 0..n_s {
                k_ss += gaussian_kernel_single(
                    &source.row(i).to_owned(),
                    &source.row(j).to_owned(),
                    sigma,
                );
            }
        }
        k_ss /= (n_s * n_s) as f64;

        // E[k(xt, xt')]
        let mut k_tt = 0.0;
        for i in 0..n_t {
            for j in 0..n_t {
                k_tt += gaussian_kernel_single(
                    &target.row(i).to_owned(),
                    &target.row(j).to_owned(),
                    sigma,
                );
            }
        }
        k_tt /= (n_t * n_t) as f64;

        // E[k(xs, xt)]
        let mut k_st = 0.0;
        for i in 0..n_s {
            for j in 0..n_t {
                k_st += gaussian_kernel_single(
                    &source.row(i).to_owned(),
                    &target.row(j).to_owned(),
                    sigma,
                );
            }
        }
        k_st /= (n_s * n_t) as f64;

        total_mmd += k_ss + k_tt - 2.0 * k_st;
    }

    total_mmd / bandwidths.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mmd_same_distribution() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let bandwidths = vec![0.5, 1.0, 2.0];
        let mmd = compute_mmd_squared(&data, &data, &bandwidths);
        assert!(mmd.abs() < 1e-10, "MMD of identical distributions should be ~0");
    }
}
```

### CORAL на Rust

```rust
use ndarray::{Array2, Axis};

/// Compute covariance matrix for a feature matrix.
///
/// # Arguments
/// * `features` - Feature matrix (n x d)
///
/// # Returns
/// Covariance matrix (d x d)
fn covariance_matrix(features: &Array2<f64>) -> Array2<f64> {
    let n = features.nrows() as f64;
    let mean = features.mean_axis(Axis(0)).unwrap();
    let centered = features - &mean.insert_axis(Axis(0));
    centered.t().dot(&centered) / (n - 1.0)
}

/// Compute CORAL loss between source and target features.
///
/// L_CORAL = (1 / 4d^2) * ||C_S - C_T||_F^2
///
/// # Arguments
/// * `source` - Source features (n_s x d)
/// * `target` - Target features (n_t x d)
///
/// # Returns
/// CORAL loss value
pub fn coral_loss(source: &Array2<f64>, target: &Array2<f64>) -> f64 {
    let d = source.ncols() as f64;

    let cov_s = covariance_matrix(source);
    let cov_t = covariance_matrix(target);

    let diff = &cov_s - &cov_t;
    let frobenius_sq: f64 = diff.iter().map(|&v| v * v).sum();

    frobenius_sq / (4.0 * d * d)
}

/// Apply CORAL transformation: transform source features to align
/// with target covariance structure.
///
/// Uses whitening + re-coloring approach:
///   1. Whiten source: x_w = C_S^{-1/2} @ (x - mu_S)
///   2. Re-color with target: x_t = C_T^{1/2} @ x_w + mu_T
pub fn coral_transform(
    source: &Array2<f64>,
    target: &Array2<f64>,
) -> Array2<f64> {
    let mean_s = source.mean_axis(Axis(0)).unwrap();
    let mean_t = target.mean_axis(Axis(0)).unwrap();

    let cov_s = covariance_matrix(source);
    let cov_t = covariance_matrix(target);

    // Center source
    let centered = source - &mean_s.insert_axis(Axis(0));

    // For simplicity, approximate with diagonal covariance
    let d = source.ncols();
    let mut result = Array2::zeros(source.raw_dim());

    for col in 0..d {
        let std_s = cov_s[[col, col]].sqrt().max(1e-8);
        let std_t = cov_t[[col, col]].sqrt().max(1e-8);
        for row in 0..source.nrows() {
            result[[row, col]] =
                centered[[row, col]] / std_s * std_t + mean_t[col];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_coral_loss_same() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let loss = coral_loss(&data, &data);
        assert!(loss.abs() < 1e-10, "CORAL loss for identical data should be ~0");
    }
}
```

### Sinkhorn Distance на Rust

```rust
use ndarray::{Array1, Array2};

/// Compute Sinkhorn approximation to Wasserstein distance.
///
/// # Arguments
/// * `source` - Source samples (n_s x d)
/// * `target` - Target samples (n_t x d)
/// * `epsilon` - Entropic regularization parameter
/// * `n_iter` - Number of Sinkhorn iterations
///
/// # Returns
/// Approximate Wasserstein distance
pub fn sinkhorn_distance(
    source: &Array2<f64>,
    target: &Array2<f64>,
    epsilon: f64,
    n_iter: usize,
) -> f64 {
    let n_s = source.nrows();
    let n_t = target.nrows();

    // Cost matrix: squared Euclidean distances
    let mut cost = Array2::zeros((n_s, n_t));
    for i in 0..n_s {
        for j in 0..n_t {
            let mut sq_dist = 0.0;
            for k in 0..source.ncols() {
                let diff = source[[i, k]] - target[[j, k]];
                sq_dist += diff * diff;
            }
            cost[[i, j]] = sq_dist;
        }
    }

    // Gibbs kernel K = exp(-C / epsilon)
    let k_matrix: Array2<f64> = cost.mapv(|c| (-c / epsilon).exp());

    // Uniform marginals
    let mu = Array1::from_elem(n_s, 1.0 / n_s as f64);
    let nu = Array1::from_elem(n_t, 1.0 / n_t as f64);

    // Sinkhorn iterations
    let mut u = Array1::ones(n_s);
    let mut v;

    for _ in 0..n_iter {
        v = &nu / &(k_matrix.t().dot(&u).mapv(|x| x.max(1e-10)));
        u = &mu / &(k_matrix.dot(&v).mapv(|x| x.max(1e-10)));
    }

    v = &nu / &(k_matrix.t().dot(&u).mapv(|x| x.max(1e-10)));

    // Compute transport plan T = diag(u) @ K @ diag(v)
    // and distance = sum(T * C)
    let mut distance = 0.0;
    for i in 0..n_s {
        for j in 0..n_t {
            let t_ij = u[i] * k_matrix[[i, j]] * v[j];
            distance += t_ij * cost[[i, j]];
        }
    }

    distance
}
```

### Cargo.toml

```toml
[package]
name = "domain-adaptation-finance"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = { version = "0.16", features = ["blas"] }
ndarray-rand = "0.15"
rand = "0.8"
polars = { version = "0.43", features = ["lazy", "csv"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.12", features = ["json", "blocking"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"

[dev-dependencies]
approx = "0.5"
```

---

## Практические примеры с данными акций и криптовалют

### Пример 1: Кросс-рыночная адаптация (S&P 500 -> Bybit BTC)

```python
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


# ===============================
# 1. Load source domain: S&P 500
# ===============================
sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']

source_data = {}
for ticker in sp500_tickers:
    df = yf.download(ticker, start='2019-01-01', end='2024-01-01')
    df.columns = [c.lower() for c in df.columns]
    df = create_trading_labels(df, forward_window=5, threshold=0.02)
    source_data[ticker] = df

source_df = pd.concat(source_data.values(), keys=source_data.keys())
print(f"Source domain (S&P 500): {len(source_df)} samples")


# ===================================
# 2. Load target domain: Bybit BTC/USDT
# ===================================
def fetch_bybit_klines(symbol='BTCUSDT', interval='D', limit=1000):
    """Fetch historical klines from Bybit API v5."""
    import requests

    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'spot',
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['retCode'] != 0:
        raise ValueError(f"Bybit API error: {data['retMsg']}")

    rows = []
    for item in data['result']['list']:
        rows.append({
            'timestamp': pd.to_datetime(int(item[0]), unit='ms'),
            'open': float(item[1]),
            'high': float(item[2]),
            'low': float(item[3]),
            'close': float(item[4]),
            'volume': float(item[5]),
        })

    df = pd.DataFrame(rows)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


btc_df = fetch_bybit_klines('BTCUSDT', interval='D', limit=1000)
btc_df = create_trading_labels(btc_df, forward_window=5, threshold=0.03)
print(f"Target domain (Bybit BTC): {len(btc_df)} samples")


# ===============================
# 3. Prepare datasets and loaders
# ===============================
source_dataset = FinancialDomainDataset(source_df, label_col='label')
target_dataset = FinancialDomainDataset(btc_df, label_col='label')

source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)


# ===============================
# 4. Train DANN
# ===============================
input_dim = source_dataset.features.shape[1]
dann_model = DANN(input_dim=input_dim, hidden_dim=128, feature_dim=64, num_classes=3)

print("Training DANN for cross-market adaptation...")
history = train_dann(
    dann_model, source_loader, target_loader,
    num_epochs=100, lr=1e-3, device='cpu'
)


# ===============================
# 5. Evaluate on target domain
# ===============================
dann_model.eval()
with torch.no_grad():
    all_preds = []
    all_labels = []
    for batch in DataLoader(target_dataset, batch_size=256):
        x, y = batch
        pred = dann_model.predict(x)
        all_preds.append(pred.argmax(dim=1))
        all_labels.append(y)

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    accuracy = (preds == labels).float().mean().item()
    print(f"Target domain accuracy (DANN): {accuracy:.4f}")
```

### Пример 2: Временная адаптация с несколькими криптовалютами Bybit

```python
# =====================================================
# Temporal Domain Adaptation: Bull Market -> Bear Market
# =====================================================

crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']


def fetch_multi_crypto(symbols, interval='D', limit=1000):
    """Fetch data for multiple crypto pairs from Bybit."""
    all_data = {}
    for symbol in symbols:
        try:
            df = fetch_bybit_klines(symbol, interval, limit)
            all_data[symbol] = df
            print(f"Fetched {symbol}: {len(df)} candles")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return all_data


crypto_data = fetch_multi_crypto(crypto_symbols)

# Split into domains by time period
# Source: bull market (2021)
# Target: bear market (2022)
source_frames = []
target_frames = []

for symbol, df in crypto_data.items():
    bull = df['2021-01-01':'2021-12-31'].copy()
    bear = df['2022-01-01':'2022-12-31'].copy()

    if len(bull) > 50:
        bull = create_trading_labels(bull, forward_window=3, threshold=0.02)
        source_frames.append(bull)
    if len(bear) > 50:
        bear = create_trading_labels(bear, forward_window=3, threshold=0.02)
        target_frames.append(bear)

source_temporal = pd.concat(source_frames)
target_temporal = pd.concat(target_frames)

print(f"Source (bull market): {len(source_temporal)} samples")
print(f"Target (bear market): {len(target_temporal)} samples")

# Compare with different adaptation methods
results = {}

# Baseline: No adaptation (train on source, evaluate on target)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

source_ds = FinancialDomainDataset(source_temporal, label_col='label')
target_ds = FinancialDomainDataset(target_temporal, label_col='label')

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5)
gb.fit(source_ds.features.numpy(), source_ds.labels.numpy())
baseline_preds = gb.predict(target_ds.features.numpy())
results['No Adaptation'] = {
    'accuracy': accuracy_score(target_ds.labels.numpy(), baseline_preds),
    'f1': f1_score(target_ds.labels.numpy(), baseline_preds, average='macro')
}

# DANN
input_dim = source_ds.features.shape[1]
src_loader = DataLoader(source_ds, batch_size=32, shuffle=True)
tgt_loader = DataLoader(target_ds, batch_size=32, shuffle=True)

dann = DANN(input_dim=input_dim, num_classes=3)
train_dann(dann, src_loader, tgt_loader, num_epochs=80, lr=5e-4)

dann.eval()
with torch.no_grad():
    preds, _, _ = dann(target_ds.features)
    dann_preds = preds.argmax(dim=1).numpy()
results['DANN'] = {
    'accuracy': accuracy_score(target_ds.labels.numpy(), dann_preds),
    'f1': f1_score(target_ds.labels.numpy(), dann_preds, average='macro')
}

# MMD
mmd_net = MMDAdaptationNetwork(input_dim=input_dim, num_classes=3)
train_mmd_network(mmd_net, src_loader, tgt_loader, num_epochs=80, mmd_weight=1.0)

mmd_net.eval()
with torch.no_grad():
    preds, _ = mmd_net(target_ds.features)
    mmd_preds = preds.argmax(dim=1).numpy()
results['MMD'] = {
    'accuracy': accuracy_score(target_ds.labels.numpy(), mmd_preds),
    'f1': f1_score(target_ds.labels.numpy(), mmd_preds, average='macro')
}

# Deep CORAL
coral_net = DeepCORAL(input_dim=input_dim, num_classes=3)
train_deep_coral(coral_net, src_loader, tgt_loader, num_epochs=80, coral_weight=1.0)

coral_net.eval()
with torch.no_grad():
    preds, _ = coral_net(target_ds.features)
    coral_preds = preds.argmax(dim=1).numpy()
results['Deep CORAL'] = {
    'accuracy': accuracy_score(target_ds.labels.numpy(), coral_preds),
    'f1': f1_score(target_ds.labels.numpy(), coral_preds, average='macro')
}

# Print comparison
print("\n" + "=" * 60)
print("Temporal Domain Adaptation Results (Bull -> Bear)")
print("=" * 60)
print(f"{'Method':<20} {'Accuracy':>10} {'F1 (macro)':>12}")
print("-" * 42)
for method, metrics in results.items():
    print(f"{method:<20} {metrics['accuracy']:>10.4f} {metrics['f1']:>12.4f}")
```

### Пример 3: Визуализация доменного сдвига с t-SNE

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_domain_shift(source_features, target_features,
                           adapted_features=None, title="Domain Shift"):
    """Visualize domain distributions using t-SNE.

    Shows source and target distributions before and after adaptation.
    """
    n_s = source_features.shape[0]
    n_t = target_features.shape[0]

    if adapted_features is not None:
        all_features = np.vstack([
            source_features, target_features, adapted_features
        ])
    else:
        all_features = np.vstack([source_features, target_features])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_features)

    fig, axes = plt.subplots(1, 2 if adapted_features is not None else 1,
                             figsize=(14, 6))

    if adapted_features is None:
        axes = [axes]

    # Before adaptation
    ax = axes[0]
    ax.scatter(embedded[:n_s, 0], embedded[:n_s, 1],
               c='blue', alpha=0.5, label='Source', s=10)
    ax.scatter(embedded[n_s:n_s+n_t, 0], embedded[n_s:n_s+n_t, 1],
               c='red', alpha=0.5, label='Target', s=10)
    ax.set_title(f'{title} - Before Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # After adaptation
    if adapted_features is not None:
        ax = axes[1]
        ax.scatter(embedded[:n_s, 0], embedded[:n_s, 1],
                   c='blue', alpha=0.5, label='Source', s=10)
        ax.scatter(embedded[n_s+n_t:, 0], embedded[n_s+n_t:, 1],
                   c='green', alpha=0.5, label='Adapted Target', s=10)
        ax.set_title(f'{title} - After Adaptation')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('domain_shift_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage with DANN
dann.eval()
with torch.no_grad():
    _, _, source_feats = dann(source_ds.features)
    _, _, target_feats = dann(target_ds.features)

visualize_domain_shift(
    source_ds.features.numpy(),
    target_ds.features.numpy(),
    target_feats.numpy(),
    title="Temporal Domain Adaptation"
)
```

---

## Фреймворк бэктестинга

### Бэктестинг с доменной адаптацией

```python
import numpy as np
import pandas as pd


class DomainAdaptationBacktester:
    """Backtesting framework for domain adaptation trading strategies.

    Supports rolling-window adaptation where the model periodically
    re-adapts to new market conditions.

    Args:
        model: Trained domain adaptation model (DANN, MMD, CORAL)
        adaptation_method: 'dann', 'mmd', 'coral', 'ot'
        initial_capital: Starting capital in USD
        commission: Trading commission per trade (fraction)
        adapt_frequency: Re-adapt every N days
        source_window: Size of source domain window (days)
        target_window: Size of target domain window (days)
    """

    def __init__(self, model, adaptation_method='dann',
                 initial_capital=100_000, commission=0.001,
                 adapt_frequency=30, source_window=252,
                 target_window=63):
        self.model = model
        self.adaptation_method = adaptation_method
        self.initial_capital = initial_capital
        self.commission = commission
        self.adapt_frequency = adapt_frequency
        self.source_window = source_window
        self.target_window = target_window

    def run(self, source_df, target_df, feature_cols, label_col='label'):
        """Run backtest on target domain data.

        Args:
            source_df: Source domain historical data
            target_df: Target domain data (backtested sequentially)
            feature_cols: List of feature column names
            label_col: Label column name

        Returns:
            Dictionary with backtest results
        """
        capital = self.initial_capital
        position = 0  # -1, 0, 1
        trades = []
        equity_curve = [capital]
        positions = []
        daily_returns = []

        target_dates = target_df.index
        scaler = StandardScaler()
        scaler.fit(source_df[feature_cols].values)

        for i in range(len(target_dates)):
            date = target_dates[i]
            row = target_df.loc[date]

            # Get features for current observation
            features = scaler.transform(
                row[feature_cols].values.reshape(1, -1)
            )
            features_tensor = torch.FloatTensor(features)

            # Predict
            self.model.eval()
            with torch.no_grad():
                if self.adaptation_method == 'dann':
                    pred = self.model.predict(features_tensor)
                else:
                    pred, _ = self.model(features_tensor)
                signal = pred.argmax(dim=1).item()

            # Map signal to position: 0=sell(-1), 1=hold(0), 2=buy(+1)
            new_position = signal - 1

            # Execute trade if position changes
            if new_position != position:
                # Close existing position
                if position != 0:
                    trade_return = position * row['close'] / entry_price - 1
                    trade_return -= self.commission * 2  # round-trip
                    capital *= (1 + trade_return)
                    trades.append({
                        'exit_date': date,
                        'exit_price': row['close'],
                        'direction': 'long' if position > 0 else 'short',
                        'return': trade_return,
                    })

                # Open new position
                if new_position != 0:
                    entry_price = row['close']
                    capital *= (1 - self.commission)

                position = new_position

            positions.append(position)
            equity_curve.append(capital)

            # Daily return
            if i > 0:
                daily_ret = equity_curve[-1] / equity_curve[-2] - 1
                daily_returns.append(daily_ret)

            # Re-adapt periodically
            if (i + 1) % self.adapt_frequency == 0 and i > self.target_window:
                self._re_adapt(source_df, target_df, i, feature_cols)

        return self._compute_metrics(
            equity_curve, daily_returns, trades, positions
        )

    def _re_adapt(self, source_df, target_df, current_idx, feature_cols):
        """Re-adapt model using recent target domain data."""
        # Get recent target data as new adaptation target
        start_idx = max(0, current_idx - self.target_window)
        recent_target = target_df.iloc[start_idx:current_idx]

        if len(recent_target) < 20:
            return

        # Quick adaptation pass (few epochs)
        target_ds = FinancialDomainDataset(recent_target)
        target_loader = DataLoader(target_ds, batch_size=32, shuffle=True)

        # Run 5 adaptation epochs
        self.model.train()
        optimizer = optim.Adam(
            self.model.feature_extractor.parameters(), lr=1e-4
        )

        for _ in range(5):
            for batch in target_loader:
                x = batch[0]
                _, features = self.model(x)
                # Mini-adaptation step (method-specific)
                optimizer.zero_grad()
                # Simplified: just ensure features stay normalized
                reg_loss = features.pow(2).mean() * 0.01
                reg_loss.backward()
                optimizer.step()

    def _compute_metrics(self, equity_curve, daily_returns, trades, positions):
        """Compute standard trading performance metrics."""
        equity = np.array(equity_curve)
        returns = np.array(daily_returns)

        total_return = equity[-1] / equity[0] - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Sortino ratio
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0
        sortino = annual_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        if trades:
            winning = [t for t in trades if t['return'] > 0]
            win_rate = len(winning) / len(trades)
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'equity_curve': equity,
            'positions': positions,
        }
```

### Запуск бэктестинга

```python
# ==============================================================
# Compare backtest performance: No adaptation vs DANN vs MMD vs CORAL
# ==============================================================

methods = {
    'DANN': dann_model,
    'MMD': mmd_net,
    'Deep CORAL': coral_net,
}

backtest_results = {}
feature_cols = [c for c in source_df.columns
                if c not in ['label', 'open', 'high', 'low', 'close', 'volume']]

for method_name, model in methods.items():
    print(f"\nBacktesting {method_name}...")
    backtester = DomainAdaptationBacktester(
        model=model,
        adaptation_method=method_name.lower().replace(' ', '_'),
        initial_capital=100_000,
        commission=0.001,
        adapt_frequency=30,
    )

    results = backtester.run(
        source_df=source_temporal,
        target_df=target_temporal,
        feature_cols=feature_cols,
    )
    backtest_results[method_name] = results


# Print results table
print("\n" + "=" * 80)
print("Backtest Results: Domain Adaptation Methods Comparison")
print("=" * 80)
print(f"{'Method':<15} {'Return':>10} {'Sharpe':>8} {'Sortino':>9} "
      f"{'MaxDD':>8} {'Trades':>8} {'WinRate':>9}")
print("-" * 67)
for method, r in backtest_results.items():
    print(f"{method:<15} {r['total_return']:>+9.2%} {r['sharpe_ratio']:>8.2f} "
          f"{r['sortino_ratio']:>9.2f} {r['max_drawdown']:>8.2%} "
          f"{r['num_trades']:>8d} {r['win_rate']:>9.2%}")
```

---

## Оценка производительности

### Метрики для оценки доменной адаптации

В доменной адаптации важно оценивать не только качество предсказаний, но и степень выравнивания доменов.

#### Метрики качества предсказаний

| Метрика | Формула | Назначение |
|---------|---------|-----------|
| **Accuracy** | $\frac{\text{TP} + \text{TN}}{N}$ | Общая точность |
| **F1 (macro)** | $\frac{1}{K}\sum_k \frac{2P_k R_k}{P_k + R_k}$ | Сбалансированная оценка по классам |
| **AUC-ROC** | Площадь под ROC-кривой | Качество ранжирования |

#### Метрики выравнивания доменов

| Метрика | Формула | Интерпретация |
|---------|---------|--------------|
| **MMD** | $\|\frac{1}{n_s}\sum \phi(x_s) - \frac{1}{n_t}\sum \phi(x_t)\|^2$ | Расхождение в RKHS ($\downarrow$ лучше) |
| **Proxy A-distance** | $2(1 - 2\epsilon_{domain})$ | Различимость доменов ($\downarrow$ лучше) |
| **CORAL distance** | $\frac{1}{4d^2}\|C_S - C_T\|_F^2$ | Расхождение ковариаций ($\downarrow$ лучше) |
| **Wasserstein** | $W_1(P_S, P_T)$ | Расстояние оптимального транспорта ($\downarrow$ лучше) |

#### Торговые метрики

| Метрика | Формула | Хороший результат |
|---------|---------|------------------|
| **Sharpe Ratio** | $\frac{R_p - R_f}{\sigma_p}$ | $> 1.0$ |
| **Sortino Ratio** | $\frac{R_p - R_f}{\sigma_{downside}}$ | $> 1.5$ |
| **Max Drawdown** | $\max_t \frac{\text{Peak}_t - \text{Value}_t}{\text{Peak}_t}$ | $< 20\%$ |
| **Calmar Ratio** | $\frac{R_{annual}}{|\text{MaxDD}|}$ | $> 1.0$ |
| **Win Rate** | $\frac{N_{winning}}{N_{total}}$ | $> 50\%$ |

### Оценка Proxy A-distance

```python
def proxy_a_distance(source_features, target_features):
    """Compute Proxy A-distance to measure domain discrepancy.

    Train a classifier to distinguish domains. If accuracy ~ 50%,
    domains are well-aligned. If accuracy ~ 100%, domains are very different.

    PAD = 2 * (1 - 2 * error)

    Returns value in [0, 2]: 0 = identical, 2 = completely different.
    """
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score

    X = np.vstack([source_features, target_features])
    y = np.hstack([
        np.zeros(len(source_features)),
        np.ones(len(target_features))
    ])

    clf = LinearSVC(max_iter=5000, dual=False)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    error = 1 - scores.mean()

    pad = 2 * (1 - 2 * error)
    return max(0, pad)  # Clamp to [0, 2]


# Example: measure alignment before and after adaptation
print("Proxy A-distance (before adaptation):")
pad_before = proxy_a_distance(
    source_ds.features.numpy(),
    target_ds.features.numpy()
)
print(f"  PAD = {pad_before:.4f}")

print("Proxy A-distance (after DANN adaptation):")
dann.eval()
with torch.no_grad():
    _, _, s_feats = dann(source_ds.features)
    _, _, t_feats = dann(target_ds.features)

pad_after = proxy_a_distance(s_feats.numpy(), t_feats.numpy())
print(f"  PAD = {pad_after:.4f}")
print(f"  Reduction: {(1 - pad_after/pad_before)*100:.1f}%")
```

### Комплексная оценка

```python
def comprehensive_evaluation(models_dict, source_ds, target_ds,
                             backtest_results):
    """Print comprehensive evaluation table for all methods.

    Args:
        models_dict: {'method_name': model}
        source_ds: Source domain dataset
        target_ds: Target domain dataset
        backtest_results: Dict of backtest results per method
    """
    print("\n" + "=" * 90)
    print("COMPREHENSIVE DOMAIN ADAPTATION EVALUATION")
    print("=" * 90)

    headers = ['Method', 'Acc', 'F1', 'PAD', 'MMD', 'Sharpe', 'MaxDD']
    print(f"{'Method':<16} {'Acc':>6} {'F1':>6} {'PAD':>6} "
          f"{'MMD':>8} {'Sharpe':>7} {'MaxDD':>7}")
    print("-" * 56)

    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'predict'):
                pred = model.predict(target_ds.features)
                _, _, s_f = model(source_ds.features)
                _, _, t_f = model(target_ds.features)
            else:
                pred, t_f = model(target_ds.features)
                _, s_f = model(source_ds.features)

            preds = pred.argmax(dim=1).numpy()
            labels = target_ds.labels.numpy()

            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            pad = proxy_a_distance(s_f.numpy(), t_f.numpy())
            mmd = compute_mmd(s_f, t_f).item()

        bt = backtest_results.get(name, {})
        sharpe = bt.get('sharpe_ratio', 0)
        maxdd = bt.get('max_drawdown', 0)

        print(f"{name:<16} {acc:>6.3f} {f1:>6.3f} {pad:>6.3f} "
              f"{mmd:>8.5f} {sharpe:>7.2f} {maxdd:>+7.2%}")
```

---

## Перспективные направления

### 1. Адаптация на основе фундаментальных моделей (Foundation Model Adaptation)

С появлением крупных фундаментальных моделей для временных рядов (TimesFM, Chronos, Lag-Llama) доменная адаптация приобретает новый контекст. Адаптация предобученных моделей к конкретному финансовому домену через Parameter-Efficient Fine-Tuning (PEFT) -- перспективное направление, позволяющее использовать общие знания модели с минимальной адаптацией.

### 2. Причинно-следственная доменная адаптация (Causal Domain Adaptation)

Стандартные методы DA выравнивают маргинальные распределения, но не гарантируют сохранение причинно-следственных связей. Каузальная доменная адаптация ищет инвариантные причинные механизмы $P(Y|X_{causal})$, которые стабильны между доменами. Это особенно релевантно для финансов, где корреляции непостоянны, но фундаментальные причинные связи (например, прибыль компании $\rightarrow$ цена акции) более устойчивы.

### 3. Непрерывная доменная адаптация (Continuous Domain Adaptation)

В реальной торговле домен смещается непрерывно, а не дискретно. Методы непрерывной адаптации (Continuous DA) моделируют домен как плавно меняющуюся среду и обновляют выравнивание в реальном времени. Это естественно сочетается с потоковым обучением (online learning) из Главы 34.

### 4. Мультисходная адаптация (Multi-Source Domain Adaptation)

Использование нескольких исходных доменов одновременно. Например, обучение на данных акций, FX и товарных рынков для создания модели, адаптируемой к любому новому рынку. Ключевая задача -- определение оптимальных весов для каждого исходного домена.

### 5. Адаптация для больших языковых моделей в трейдинге

Адаптация финансовых LLM (FinGPT, BloombergGPT) к специфическим рынкам и стилям торговли через доменно-адаптивное fine-tuning. Это включает выравнивание представлений финансовых текстов между различными языками, рынками и типами документов.

### 6. Федеративная доменная адаптация

Доменная адаптация в федеративной среде, где данные из разных доменов (бирж, фондов) не могут быть объединены из-за ограничений конфиденциальности. Модель адаптируется к целевому домену, используя только агрегированные статистики или градиенты из исходных доменов.

---

## Литература

### Основополагающие работы

1. **Ben-David, S., Blitzer, J., Crammer, K., Kuber, A., Pereira, F., & Vaughan, J. W.** (2010). A theory of learning from different domains. *Machine Learning*, 79(1-2), 151-175.

2. **Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V.** (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1-35.

3. **Sun, B., & Saenko, K.** (2016). Deep CORAL: Correlation alignment for deep domain adaptation. In *ECCV Workshops* (pp. 443-450).

4. **Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B., & Smola, A.** (2012). A kernel two-sample test. *Journal of Machine Learning Research*, 13, 723-773.

5. **Courty, N., Flamary, R., Tuia, D., & Rakotomamonjy, A.** (2017). Optimal transport for domain adaptation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(9), 1853-1865.

### Финансовые приложения

6. **Du, Y., Wang, J., Feng, W., Pan, S., Qin, T., Xu, R., & Wang, D.** (2024). AdaRNN: Adaptive learning and forecasting of time series. In *CIKM* (pp. 402-411).

7. **Liu, X., Xia, J., Yu, J., Shen, L., & Wang, J.** (2022). Domain adaptation for time series forecasting via attention sharing. *arXiv preprint arXiv:2203.12501*.

8. **Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G.** (2018). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

### Связанные главы

- [Глава 17: Глубокое обучение](../17_deep_learning) -- основы нейронных сетей
- [Глава 22: Глубокое обучение с подкреплением](../22_deep_reinforcement_learning) -- адаптивные агенты
- [Глава 28: Обнаружение режимов с HMM](../28_regime_detection_hmm) -- детекция смены режимов
- [Глава 34: Онлайн-обучение](../34_online_learning_adaptive) -- непрерывная адаптация
- [Глава 61: FinGPT](../61_fingpt_financial_llm) -- фундаментальные модели для финансов
