from __future__ import annotations

import os

from PyQt6 import QtCore, QtWidgets

from anylabeling.services.remote_storage import RemoteStorageClient


class _RegistryOperation(QtCore.QObject):
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.completed.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class ServerModelRegistryDialog(QtWidgets.QDialog):
    """Browse curated server models and run privileged install actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Server Model Registry"))
        self.resize(1000, 520)
        self.client = RemoteStorageClient()
        self.models = []
        self.thread = None
        self.worker = None

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Model"),
                self.tr("Source"),
                self.tr("Installed"),
                self.tr("Enabled"),
                self.tr("Loaded"),
                self.tr("Registered"),
                self.tr("Configuration"),
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.itemSelectionChanged.connect(self._update_actions)

        self.status_label = QtWidgets.QLabel()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        self.refresh_button = QtWidgets.QPushButton(self.tr("Refresh"))
        self.install_button = QtWidgets.QPushButton(
            self.tr("Install on Server")
        )
        self.enable_button = QtWidgets.QPushButton(
            self.tr("Enable after Restart")
        )
        self.close_button = QtWidgets.QPushButton(self.tr("Close"))

        self.refresh_button.clicked.connect(self.refresh)
        self.install_button.clicked.connect(self.install_selected)
        self.enable_button.clicked.connect(self.enable_selected)
        self.close_button.clicked.connect(self.accept)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.status_label, 1)
        buttons.addWidget(self.progress)
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self.install_button)
        buttons.addWidget(self.enable_button)
        buttons.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                self.tr(
                    "Models are downloaded and stored on the inference server. "
                    "Ordinary users can browse and use enabled models."
                )
            )
        )
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)

        self.refresh()

    @staticmethod
    def _yes_no(value):
        return QtWidgets.QTableWidgetItem(
            QtCore.QCoreApplication.translate(
                "ServerModelRegistryDialog", "Yes" if value else "No"
            )
        )

    def refresh(self):
        try:
            self.models = self.client.list_model_registry()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, self.tr("Model Registry Error"), str(exc)
            )
            self.models = []

        self.table.setRowCount(len(self.models))
        for row, model in enumerate(self.models):
            name = QtWidgets.QTableWidgetItem(
                model.get("display_name") or model["model_id"]
            )
            name.setData(QtCore.Qt.ItemDataRole.UserRole, model["model_id"])
            source = model.get("source") or {}
            source_text = (
                f"{source.get('provider', '')}: {source.get('model_id', '')}"
            ).strip(": ")
            values = [
                name,
                QtWidgets.QTableWidgetItem(source_text),
                self._yes_no(model.get("installed")),
                self._yes_no(model.get("enabled")),
                self._yes_no(model.get("loaded")),
                self._yes_no(model.get("registered")),
                self._yes_no(model.get("config_available")),
            ]
            for column, item in enumerate(values):
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.status_label.setText(
            self.tr("%d registry models") % len(self.models)
        )
        self._update_actions()

    def _selected_model(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.models):
            return None
        return self.models[row]

    def _update_actions(self):
        model = self._selected_model()
        is_admin = bool(
            os.getenv("XANYLABELING_MODEL_UPLOAD_API_KEY", "")
        )
        idle = self.thread is None
        self.refresh_button.setEnabled(idle)
        self.close_button.setEnabled(idle)
        self.install_button.setEnabled(
            bool(model and not model.get("installed") and is_admin and idle)
        )
        self.enable_button.setEnabled(
            bool(
                model
                and model.get("installed")
                and not model.get("enabled")
                and model.get("registered")
                and model.get("config_available")
                and is_admin
                and idle
            )
        )

    def install_selected(self):
        model = self._selected_model()
        if model:
            self._run_operation(
                lambda: self.client.install_registry_model(model["model_id"]),
                self.tr("Downloading model on the server..."),
            )

    def enable_selected(self):
        model = self._selected_model()
        if model:
            self._run_operation(
                lambda: self.client.enable_registry_model(model["model_id"]),
                self.tr("Enabling model for the next service restart..."),
            )

    def _run_operation(self, operation, status):
        self.status_label.setText(status)
        self.progress.show()
        self.thread = QtCore.QThread(self)
        self.worker = _RegistryOperation(operation)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._operation_completed)
        self.worker.failed.connect(self._operation_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._operation_finished)
        self.thread.start()
        self._update_actions()

    def _operation_completed(self, result):
        message = self.tr("Server operation completed.")
        if result.get("restart_required"):
            message += "\n" + self.tr(
                "Restart langgao-autolabel.service to load the model."
            )
        QtWidgets.QMessageBox.information(
            self, self.tr("Model Registry"), message
        )

    def _operation_failed(self, message):
        QtWidgets.QMessageBox.critical(
            self, self.tr("Model Registry Error"), message
        )

    def _operation_finished(self):
        self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.progress.hide()
        self.refresh()

    def reject(self):
        if self.thread is not None:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Model Registry"),
                self.tr("Wait for the server operation to finish."),
            )
            return
        super().reject()
