"""Cell-based mesh visualization canvas."""

from __future__ import annotations

import math

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from meshcfd.core.model import FiniteVolumeMesh, InitialState

PATCH_COLORS = (
    QColor("#2563EB"),
    QColor("#16A34A"),
    QColor("#EA580C"),
    QColor("#9333EA"),
    QColor("#0891B2"),
    QColor("#DC2626"),
)
COLOR_MAPS = {
    "Viridis": ("#440154", "#3B528B", "#21918C", "#5EC962", "#FDE725"),
    "Plasma": ("#0D0887", "#7E03A8", "#CC4778", "#F89540", "#F0F921"),
    "Inferno": ("#000004", "#420A68", "#932667", "#DD513A", "#FBA40A", "#FCFFA4"),
    "Coolwarm": ("#3B4CC0", "#8DB0FE", "#F7F7F7", "#F4987A", "#B40426"),
}
SOLUTION_VARIABLES = {
    "Density": "density",
    "Velocity X": "velocity_x",
    "Velocity Y": "velocity_y",
    "Pressure": "pressure",
}


def _palette_color(position: float, palette: str) -> QColor:
    stops = COLOR_MAPS[palette]
    scaled = max(0.0, min(1.0, position)) * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    fraction = scaled - index
    first = QColor(stops[index])
    second = QColor(stops[index + 1])
    return QColor(
        round(first.red() + fraction * (second.red() - first.red())),
        round(first.green() + fraction * (second.green() - first.green())),
        round(first.blue() + fraction * (second.blue() - first.blue())),
    )


def color_values(values: tuple[float, ...], palette: str, scale: str) -> tuple[QColor, ...]:
    if not values:
        return ()
    if scale == "Logarithmic":
        if any(value <= 0.0 for value in values):
            raise ValueError("Logarithmic color scale requires all values to be positive.")
        values = tuple(math.log10(value) for value in values)
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if not math.isfinite(span):
        magnitude = max(abs(minimum), abs(maximum))
        values = tuple(value / magnitude for value in values)
        minimum = min(values)
        maximum = max(values)
        span = maximum - minimum
    return tuple(
        _palette_color(0.5 if span == 0.0 else (value - minimum) / span, palette)
        for value in values
    )


class ColorLegend(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.palette = "Viridis"
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.setMinimumHeight(42)

    def set_values(self, values: tuple[float, ...] | None, palette: str) -> None:
        self.palette = palette
        self.minimum = min(values) if values else None
        self.maximum = max(values) if values else None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        bar = self.rect().adjusted(34, 3, -34, -18)
        gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        stops = COLOR_MAPS[self.palette]
        for index, stop in enumerate(stops):
            gradient.setColorAt(index / (len(stops) - 1), QColor(stop))
        painter.fillRect(bar, gradient)
        painter.setPen(QColor("#475569"))
        if self.minimum is None or self.maximum is None:
            low = high = "—"
        else:
            low = f"{self.minimum:.4g}"
            high = f"{self.maximum:.4g}"
        painter.drawText(0, self.height() - 2, low)
        painter.drawText(
            self.width() - painter.fontMetrics().horizontalAdvance(high),
            self.height() - 2,
            high,
        )
        event.accept()


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
        self.solution: tuple[InitialState, ...] | None = None
        self.show_grid = True
        self.show_mesh = True
        self.solution_variable = "Density"
        self.color_map = "Viridis"
        self.color_scale = "Linear"
        self._panning = False
        self._last_mouse_pos = None
        self._show_empty_message()

    def _show_empty_message(self) -> None:
        self.scene().clear()

    def drawForeground(self, painter, rect) -> None:
        if self.mesh is None:
            painter.save()
            painter.resetTransform()
            painter.setClipRect(rect, Qt.IntersectClip)
            painter.setPen(QColor("#64748B"))
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignCenter,
                "Import a MeshCAD finite-volume mesh JSON",
            )
            painter.restore()

    def drawBackground(self, painter, rect) -> None:
        super().drawBackground(painter, rect)
        if not self.show_grid:
            return
        transform = self.transform()
        zoom = max(abs(transform.m11()), abs(transform.m22()), 1e-12)
        target_spacing = 80.0 / zoom
        exponent = math.floor(math.log10(target_spacing))
        base = 10.0**exponent
        candidates = (base, 2.0 * base, 5.0 * base, 10.0 * base)
        spacing = next(
            (candidate for candidate in candidates if candidate * zoom >= 60.0),
            10.0 * base,
        )
        min_x = math.floor(rect.left() / spacing) * spacing
        max_x = math.ceil(rect.right() / spacing) * spacing
        min_y = math.floor(rect.top() / spacing) * spacing
        max_y = math.ceil(rect.bottom() / spacing) * spacing
        if (max_x - min_x) / spacing > 250 or (max_y - min_y) / spacing > 250:
            return
        pen = QPen(QColor("#E6E6EB"), 0.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = min_x
        while x <= max_x:
            painter.drawLine(QPointF(x, min_y), QPointF(x, max_y))
            x += spacing
        y = min_y
        while y <= max_y:
            painter.drawLine(QPointF(min_x, y), QPointF(max_x, y))
            y += spacing

    def set_mesh(self, mesh: FiniteVolumeMesh | None) -> None:
        self.mesh = mesh
        self.selected_patch_id = None
        self.solution = None
        self._draw_mesh(fit=True)

    def set_solution(self, solution: tuple[InitialState, ...] | None) -> None:
        self.solution = solution

    def set_display_options(
        self,
        *,
        show_grid: bool,
        show_mesh: bool,
        solution_variable: str,
        color_map: str,
        color_scale: str,
    ) -> tuple[float, ...]:
        values = self._solution_values(solution_variable)
        colors = color_values(values, color_map, color_scale)
        self.show_grid = show_grid
        self.show_mesh = show_mesh
        self.solution_variable = solution_variable
        self.color_map = color_map
        self.color_scale = color_scale
        self._draw_mesh()
        self.viewport().update()
        return values if colors else ()

    def _solution_values(self, variable: str) -> tuple[float, ...]:
        if self.mesh is None or self.solution is None:
            return ()
        if len(self.solution) != len(self.mesh.cells):
            raise ValueError("Initial solution does not match the loaded mesh.")
        field = SOLUTION_VARIABLES[variable]
        return tuple(getattr(state, field) for state in self.solution)

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

        values = self._solution_values(self.solution_variable)
        colors = color_values(values, self.color_map, self.color_scale)
        for cell_id, cell in enumerate(self.mesh.cells):
            polygon = QPolygonF()
            for node_id in cell:
                x, y = self.mesh.points[node_id]
                polygon.append(QPointF(x, y))
            item = QGraphicsPolygonItem(polygon)
            if self.show_mesh:
                pen = QPen(QColor("#94A3B8"), 0.7)
                pen.setCosmetic(True)
                item.setPen(pen)
            else:
                item.setPen(QPen(Qt.NoPen))
            fill = colors[cell_id] if colors else QColor("#F1F5F9")
            item.setBrush(QBrush(fill))
            self.scene().addItem(item)

        if not self.show_mesh:
            bounds = self.scene().itemsBoundingRect()
            margin = max(1.0, max(bounds.width(), bounds.height()) * 0.04)
            bounds.adjust(-margin, -margin, margin, margin)
            self.scene().setSceneRect(bounds)
            if fit:
                self.fitInView(bounds, Qt.KeepAspectRatio)
            return

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
