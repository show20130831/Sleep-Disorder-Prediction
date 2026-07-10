import json

import matplotlib.pyplot as plt
import pandas as pd

from sleep_disorder.utils import ensure_dir, save_figure, save_json, save_table


def test_ensure_dir_creates_nested_path(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.exists()
    assert result == target


def test_save_table_creates_parent_dir_and_writes_csv(tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    path = tmp_path / "nested" / "table.csv"
    save_table(df, path)

    assert path.exists()
    loaded = pd.read_csv(path)
    assert loaded.equals(df)


def test_save_json_creates_parent_dir_and_writes_readable_json(tmp_path):
    data = {"a": 1, "b": [1, 2, 3]}
    path = tmp_path / "nested" / "data.json"
    save_json(data, path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_save_figure_creates_parent_dir_and_writes_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "nested" / "figure.png"

    save_figure(fig, path)

    assert path.exists()
    assert path.stat().st_size > 0
