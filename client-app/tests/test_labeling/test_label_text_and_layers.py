import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from anylabeling.views.labeling.label_widget import _reorder_layer_items
from anylabeling.views.labeling.widgets.unique_label_qlist_widget import (
    UniqueLabelQListWidget,
)


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_reorder_layer_items():
    items = [object() for _ in range(5)]
    selected = [items[1], items[2]]

    assert _reorder_layer_items(items, selected, "top") == [
        items[0],
        items[3],
        items[4],
        items[1],
        items[2],
    ]
    assert _reorder_layer_items(items, selected, "bottom") == [
        items[1],
        items[2],
        items[0],
        items[3],
        items[4],
    ]
    assert _reorder_layer_items(items, selected, "up") == [
        items[0],
        items[3],
        items[1],
        items[2],
        items[4],
    ]
    assert _reorder_layer_items(items, selected, "down") == [
        items[1],
        items[2],
        items[0],
        items[3],
        items[4],
    ]


def test_unique_label_checkbox_emits_text_visibility():
    _app()
    widget = UniqueLabelQListWidget()
    item = widget.create_item_from_label("normal_contact")
    widget.addItem(item)
    widget.set_item_label(item, "normal_contact", (100, 150, 200), 128)

    emitted = []
    widget.label_text_visibility_changed.connect(
        lambda label, visible: emitted.append((label, visible))
    )
    item.setCheckState(Qt.CheckState.Unchecked)

    assert emitted[-1] == ("normal_contact", False)
