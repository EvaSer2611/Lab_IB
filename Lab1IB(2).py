import math

# --- Типы защиты окон ---
window_protection_types = {
    1: {"name": "Открытое окно", "probability": 1.0},
    2: {"name": "Решетки простые", "probability": 0.7},
    3: {"name": "Армированное стекло", "probability": 0.4},
    4: {"name": "Рольставни", "probability": 0.2},
    5: {"name": "Забетонированное окно", "probability": 0.0},
    
}

def choose_protection(prompt):
    """Просит пользователя выбрать тип защиты (1..5) и возвращает вероятность."""
    print(prompt)
    for k, v in window_protection_types.items():
        print(f"  {k}: {v['name']} (prob={v['probability']})")
    while True:
        try:
            choice = int(input("Введите номер типа защиты (1-5): ").strip())
            if choice in window_protection_types:
                return window_protection_types[choice]["probability"]
            else:
                print("Ошибка: введите число от 1 до 5.")
        except ValueError:
            print("Ошибка: введите корректное целое число.")

# --- Входные данные (ваши) ---
Xzab = 50
Xok1 = 20
Xok2 = 30
Yzab = 40
Yokn = 25
Xinf = 10
Yinf = 10
k1 = 2.0
k2 = 0.5

# Пользователь выбирает тип защиты для окна1 и окна2
print("Выберите тип защиты для первого окна:")
prot_prob1 = choose_protection("Первое окно:")
print("\nВыберите тип защиты для второго окна:")
prot_prob2 = choose_protection("Второе окно:")
print(f"\nВыбрано: окно1 prob={prot_prob1}, окно2 prob={prot_prob2}\n")

# Количество участков (0 < i < 100 => возьмём 100 участков)
n = 100

eps = 1e-12  # защита от деления на ноль

# заранее посчитаем расстояния "окно -> инф" (оно не зависит от i)
l21 = math.hypot(Xok1 - Xinf, Yokn - Yinf)  # окно1 -> инф
l22 = math.hypot(Xok2 - Xinf, Yokn - Yinf)  # окно2 -> инф

# убедимся, что не ноль
l21 = max(l21, eps)
l22 = max(l22, eps)

p1 = []  # список кортежей (x, Pi1)
p2 = []  # список кортежей (x, Pi2)

for i in range(n + 1):   # индексы 0..n (получим n+1 точек включая концы)
    x = i * Xzab / n
    y = Yzab

    # расстояния fence -> window1 и fence -> window2 (зависят от x)
    l11 = math.hypot(x - Xok1, y - Yokn)
    l12 = math.hypot(x - Xok2, y - Yokn)

    l11 = max(l11, eps)
    l12 = max(l12, eps)

    # P_ok вычисляем по вашей инструкции (k1 / L_ab)
    # интерпретация: используем L_ab = расстояние окно -> инф (l21 или l22)
    P_ok1 = k1 / l21
    P_ok2 = k1 / l22

    # итоговые вероятности для текущего участка i:
    # добавляем множитель защиты окна (prot_prob1 / prot_prob2)
    Pi1 = (k1 / l11) * P_ok1 * (k2 / l21) * prot_prob1
    Pi2 = (k1 / l12) * P_ok2 * (k2 / l22) * prot_prob2

    p1.append((x, Pi1))
    p2.append((x, Pi2))

# Печать нескольких результатов для контроля (покажем первые 6 значений для наглядности)
print("50 значений Pi по окну1 (x, Pi1):")
for t in p1[:51]:
    print(t)

print("\n50 значений Pi по окну2 (x, Pi2):")
for t in p2[:51]:
    print(t)

# Минимумы
min1 = min(p1, key=lambda item: item[1])
min2 = min(p2, key=lambda item: item[1])
all_vals = [v for _, v in p1] + [v for _, v in p2]
global_min = min(all_vals) if all_vals else None

print("\nМинимум через окно1:", min1)
print("Минимум через окно2:", min2)
print("Глобальный минимум P:", global_min)
