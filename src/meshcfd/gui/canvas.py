"""Cell-based mesh visualization canvas."""

from __future__ import annotations

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
)

from meshcfd.core.model import FiniteVolumeMesh

PATCH_COLORS = (
    QColor("#2563EB"),
    QColor("#16A34A"),
    QColor("#EA580C"),
    QColor("#9333EA"),
    QColor("#0891B2"),
    QColor("#DC2626"),
)


class MeshCanvas(QGraphicsView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#FAFAFC")))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(1.0, -1.0)
        self.mesh: FiniteVolumeMesh | None = None
        self.selected_patch_id: int | None = None
        self._panning = False
        self._last_mouse_pos = None
        self._show_empty_message()

    def _show_empty_message(self) -> None:
        self.scene().clear()

    def drawForeground(self, painter, rect) -> None:
        if self.mesh is None:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor("#64748B"))
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignCenter,
                "Import a MeshCAD finite-volume mesh JSON",
            )
            painter.restore()

    def set_mesh(self, mesh: FiniteVolumeMesh | None) -> None:
        self.mesh = mesh
        self.selected_patch_id = None
        self._draw_mesh(fit=True)

    def select_patch(self, patch_id: int | None) -> None:
        if self.selected_patch_id == patch_id:
            return
        self.selected_patch_id = patch_id
        self._draw_mesh()

    def _draw_mesh(self, fit: bool = False) -> None:
        self.scene().clear()
        if self.mesh is None:
            self._show_empty_message()
            return

        for cell in self.mesh.cells:
            polygon = QPolygonF()
            for node_id in cell:
                x, y = self.mesh.points[node_id]
                polygon.append(QPointF(x, y))
            item = QGraphicsPolygonItem(polygon)
            pen = QPen(QColor("#94A3B8"), 0.7)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(QColor("#F1F5F9")))
            self.scene().addItem(item)

        for face_id, (start_id, end_id) in enumerate(self.mesh.faces):
            x0, y0 = self.mesh.points[start_id]
            x1, y1 = self.mesh.points[end_id]
            patch_id = self.mesh.face_patch_ids[face_id]
            is_boundary = patch_id >= 0
            selected = patch_id == self.selected_patch_id
            if is_boundary:
                patch_index = next(
                    index
                    for index, patch in enumerate(self.mesh.patches)
                    if patch.id == patch_id
                )
                color = PATCH_COLORS[patch_index % len(PATCH_COLORS)]
                width = 3.0 if selected else 1.6
            else:
                color = QColor("#CBD5E1")
                width = 0.7
            pen = QPen(color, width)
            pen.setCosmetic(True)
            item = QGraphicsLineItem(x0, y0, x1, y1)
            item.setPen(pen)
            self.scene().addItem(item)

        bounds = self.scene().itemsBoundingRect()
        if bounds.width() < 1e-9:
            bounds.adjust(-0.5, 0.0, 0.5, 0.0)
        if bounds.height() < 1e-9:
            bounds.adjust(0.0, -0.5, 0.0, 0.5)
        margin = max(1.0, max(bounds.width(), bounds.height()) * 0.04)
        bounds.adjust(-margin, -margin, margin, margin)
        self.scene().setSceneRect(bounds)
        if fit:
            self.fitInView(bounds, Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        if self.mesh is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
