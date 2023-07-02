# MISIS DS.HACK 

Решения хакатона по анализу данных.

## Setup
Требуется Python >= 3.11 и cuda для ноутбука с NN моделью
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```
## Струкутура проекта
Исходный код проекта (`.py` файлы) находится в папке `./src`.

### Задачи по notebooks

1) Основная задача, риски: `./notebooks/data_analysis.ipynb`
2) Построение и оценка инвестиционного портфеля: `./notebooks/portfolio_selection.ipynb`
3) Прогнозирование: `./notebooks/stocks_prices_prediction.ipynb`
