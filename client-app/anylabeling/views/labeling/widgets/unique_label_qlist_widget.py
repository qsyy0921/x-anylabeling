# -*- encoding: utf-8 -*-

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    label_text_visibility_changed = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label_text_visibility = {}
        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item):
        label = item.data(Qt.ItemDataRole.UserRole)
        if label is None:
            return
        visible = item.checkState() == Qt.CheckState.Checked
        self._label_text_visibility[label] = visible
        self.label_text_visibility_changed.emit(label, visible)

    # QT Overload
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if not self.indexAt(event.position().toPoint()).isValid():
            self.clearSelection()

    def find_items_by_label(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == label:
                items.append(item)
        return items

    def create_item_from_label(self, label):
        item = QtWidgets.QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked
            if self._label_text_visibility.get(label, True)
            else Qt.CheckState.Unchecked
        )
        item.setToolTip(
            self.tr("Checked: show text for this label class")
        )
        return item

    def set_item_label(self, item, label, color=None, opacity=255):
        item.setText(label)
        if color is not None:
            background_color = QtGui.QColor(*color, opacity)
            item.setBackground(QtGui.QBrush(background_color))
        item.setSizeHint(QtCore.QSize(0, max(28, item.sizeHint().height())))

    def update_item_color(self, label, color, opacity=255):
        items = self.find_items_by_label(label)
        for item in items:
            background_color = QtGui.QColor(*color, opacity)
            item.setBackground(QtGui.QBrush(background_color))
            break

    def remove_items_by_label(self, label):
        items = self.find_items_by_label(label)
        for item in items:
            row = self.row(item)
            self.takeItem(row)
