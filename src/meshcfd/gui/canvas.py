"""Cell-based mesh visualization canvas."""

from __future__ import annotations

import math

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QRubberBand,
    QSizePolicy,
    QVBoxLayout,
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
    "Rainbow": (
        "#6E40AA",
        "#417DE0",
        "#1AC7C2",
        "#7AD151",
        "#F9E721",
        "#F89B23",
        "#D7191C",
    ),
    "Turbo": (
        "#30123B",
        "#4662D7",
        "#35ABFC",
        "#1AE4B6",
        "#72FE5E",
        "#C8EF34",
        "#FBB938",
        "#F66B19",
        "#C6230C",
        "#7A0403",
    ),
    "Cividis": ("#00204C", "#414D6B", "#7C7B78", "#BCA574", "#FFE945"),
}
SOLUTION_VARIABLES = {
    "Density": "density",
    "Velocity X": "velocity_x",
    "Velocity Y": "velocity_y",
    "Pressure": "pressure",
    "Turbulent k": "turbulence_k",
    "Specific dissipation rate": "turbulence_omega",
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


def color_values(
    values: tuple[float, ...],
    palette: str,
    scale: str,
    value_range: tuple[float, float] | None = None,
) -> tuple[QColor, ...]:
    if not values and value_range is None:
        return ()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Color values must be finite.")
    if value_range is None and values:
        minimum, maximum = min(values), max(values)
    else:
        minimum, maximum = value_range or (0.0, 0.0)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or (value_range is not None and minimum >= maximum)
    ):
        raise ValueError("Color range minimum must be finite and less than its maximum.")
    if scale == "Logarithmic":
        if minimum <= 0.0 or any(value <= 0.0 for value in values):
            raise ValueError(
                "Logarithmic color scale requires the range and all values to be positive."
            )
    if not values:
        return ()
    if scale == "Logarithmic":
        values = tuple(math.log10(value) for value in values)
        minimum, maximum = math.log10(minimum), math.log10(maximum)
    span = maximum - minimum
    scaled_values = values
    if not math.isfinite(span):
        magnitude = max(abs(minimum), abs(maximum))
        minimum, maximum = minimum / magnitude, maximum / magnitude
        scaled_values = tuple(value / magnitude for value in values)
        span = maximum - minimum
    return tuple(
        _palette_color(
            0.5 if span == 0.0 else (value - minimum) / span,
            palette,
        )
        for value in scaled_values
    )


class ColorLegend(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.palette = "Viridis"
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.scale = "Linear"
        self.orientation = "Horizontal"
        self.variable = ""
        self.setMinimumHeight(42)

    def set_values(
        self,
        values: tuple[float, ...] | None,
        palette: str,
        scale: str = "Linear",
        value_range: tuple[float, float] | None = None,
        orientation: str = "Horizontal",
        variable: str = "",
    ) -> None:
        self.palette = palette
        self.scale = scale
        self.orientation = orientation
        self.variable = variable
        if value_range is not None:
            self.minimum, self.maximum = value_range
        elif values:
            self.minimum, self.maximum = min(values), max(values)
        else:
            self.minimum = self.maximum = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        vertical = self.orientation == "Vertical"
        if vertical:
            bar = self.rect().adjusted(8, 28, -62, -8)
            gradient = QLinearGradient(0, bar.bottom(), 0, bar.top())
        else:
            bar = self.rect().adjusted(28, 24, -28, -20)
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        stops = COLOR_MAPS[self.palette]
        for index, stop in enumerate(stops):
            gradient.setColorAt(index / (len(stops) - 1), QColor(stop))
        painter.fillRect(bar, gradient)
        painter.setPen(QColor("#475569"))
        if self.minimum is None or self.maximum is None:
            ticks: tuple[float, ...] = ()
        else:
            if self.scale == "Logarithmic":
                low_log, high_log = math.log10(self.minimum), math.log10(self.maximum)
                ticks = tuple(
                    10 ** (low_log + (high_log - low_log) * fraction)
                    for fraction in (0.0, 0.5, 1.0)
                )
            else:
                ticks = (
                    self.minimum,
                    self.minimum / 2 + self.maximum / 2,
                    self.maximum,
                )
        if not ticks:
            painter.drawText(bar, Qt.AlignCenter, "No solution")
        else:
            metrics = painter.fontMetrics()
            if vertical:
                vertical_ticks = tuple(reversed(ticks))
                for fraction, value in zip((0.0, 0.5, 1.0), vertical_ticks):
                    position = round(bar.top() + fraction * bar.height())
                    painter.drawLine(bar.right(), position, bar.right() + 3, position)
                    label = f"{value:.4g}"
                    painter.drawText(
                        bar.right() + 7,
                        position + metrics.ascent() // 2,
                        label,
                    )
            else:
                for fraction, value in zip((0.0, 0.5, 1.0), ticks):
                    position = round(bar.left() + fraction * bar.width())
                    painter.drawLine(position, bar.bottom(), position, bar.bottom() + 3)
                    label = f"{value:.4g}"
                    label_width = metrics.horizontalAdvance(label)
                    left = max(0, min(self.width() - label_width, position - label_width // 2))
                    painter.drawText(left, self.height() - 2, label)
        if self.variable:
            painter.setPen(QColor("#334155"))
            title_rect = (
                self.rect().adjusted(2, 1, -2, -self.height() + 24)
                if vertical
                else self.rect().adjusted(2, 1, -2, -self.height() + 20)
            )
            painter.drawText(
                title_rect,
                Qt.AlignCenter,
                self.variable,
            )
        event.accept()


class _ColorBarResizeHandle(QWidget):
    def __init__(self, overlay: ColorBarOverlay) -> None:
        super().__init__(overlay)
        self.overlay = overlay
        self._start_position: QPoint | None = None
        self._start_size: QSize | None = None
        self.setFixedSize(16, 16)
        self.setCursor(Qt.SizeFDiagCursor)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#64748B"), 1))
        for offset in (5, 9, 13):
            painter.drawLine(self.width() - offset, self.height(), self.width(), self.height() - offset)
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._start_position = event.globalPos()
            self._start_size = self.overlay.size()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._start_position is not None
            and self._start_size is not None
            and event.buttons() & Qt.LeftButton
        ):
            delta = event.globalPos() - self._start_position
            self.overlay.resize(
                max(self.overlay.minimumWidth(), self._start_size.width() + delta.x()),
                max(self.overlay.minimumHeight(), self._start_size.height() + delta.y()),
            )
            self.overlay.move_to(self.overlay.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._start_position = None
        super().mouseReleaseEvent(event)


class ColorBarOverlay(QFrame):
    def __init__(self, canvas: MeshCanvas) -> None:
        super().__init__(canvas.viewport())
        self.canvas = canvas
        self._drag_offset: QPoint | None = None
        self.orientation = "Horizontal"
        self.setObjectName("color_bar_overlay")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(82, 82)
        self.resize(310, 72)
        self.setStyleSheet(
            "#color_bar_overlay { background: #FFFFFF; border: 1px solid #94A3B8; }"
        )
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        self.legend = ColorLegend(self)
        self.resize_handle = _ColorBarResizeHandle(self)
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.addStretch(1)
        legend_row.addWidget(self.resize_handle, 0, Qt.AlignBottom | Qt.AlignRight)
        layout.addWidget(self.legend, 1)
        layout.addLayout(legend_row)
        self.legend.installEventFilter(self)
        self.legend.setCursor(Qt.SizeAllCursor)
        self.move(16, 16)
        self.hide()

    def set_values(
        self,
        values: tuple[float, ...] | None,
        palette: str,
        scale: str,
        value_range: tuple[float, float] | None,
        orientation: str,
        variable: str,
    ) -> None:
        self.legend.set_values(
            values, palette, scale, value_range, orientation, variable
        )
        if orientation != self.orientation:
            self.orientation = orientation
            if orientation == "Vertical":
                self.setMinimumSize(135, 170)
                self.resize(150, 250)
            else:
                self.setMinimumSize(250, 62)
                self.resize(310, 72)
            self.move_to(self.pos())

    def move_to(self, position: QPoint) -> None:
        bounds = self.parentWidget().rect()
        x = max(0, min(position.x(), max(0, bounds.width() - self.width())))
        y = max(0, min(position.y(), max(0, bounds.height() - self.height())))
        self.move(x, y)

    def eventFilter(self, watched, event) -> bool:
        if watched is not self.legend:
            return super().eventFilter(watched, event)
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._drag_offset = (
                event.globalPos() - self.mapToGlobal(QPoint(0, 0))
            )
            return True
        if event.type() == QEvent.MouseMove and self._drag_offset is not None:
            position = (
                self.parentWidget().mapFromGlobal(event.globalPos())
                - self._drag_offset
            )
            self.move_to(position)
            return True
        if event.type() == QEvent.MouseButtonRelease:
            self._drag_offset = None
            return True
        return super().eventFilter(watched, event)


class MeshCanvas(QGraphicsView):
    cell_selected = pyqtSignal(int, float, float, str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
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
        self.color_range: tuple[float, float] | None = None
        self.smooth_colors = False
        self.show_colorbar = False
        self.interaction_mode = "Navigate"
        self._panning = False
        self._last_mouse_pos = None
        self._zoom_start: QPoint | None = None
        self._zoom_rubber_band = QRubberBand(
            QRubberBand.Rectangle, self.viewport()
        )
        self._cell_bounds: tuple[tuple[float, float, float, float], ...] = ()
        self._cell_centers: tuple[tuple[float, float], ...] = ()
        self._smooth_cache_key = None
        self._smooth_cache: tuple[QImage, QRectF] | None = None
        self.color_bar_overlay = ColorBarOverlay(self)
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
        self._draw_coordinate_labels(painter, rect, spacing, min_x, max_x, min_y, max_y)

    def _draw_coordinate_labels(
        self,
        painter: QPainter,
        rect,
        spacing: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> None:
        viewport = self.viewport().rect()
        painter.save()
        painter.resetTransform()
        painter.setClipRect(viewport, Qt.IntersectClip)
        painter.setPen(QColor("#475569"))
        background = QColor("#FAFAFC")
        background.setAlpha(225)
        x = min_x
        while x <= max_x:
            point = self.mapFromScene(QPointF(x, rect.center().y()))
            label = f"{x:.6g}"
            self._draw_coordinate_label(painter, label, point.x(), viewport.bottom() - 3, background)
            x += spacing
        y = min_y
        while y <= max_y:
            point = self.mapFromScene(QPointF(rect.center().x(), y))
            label = f"{y:.6g}"
            self._draw_coordinate_label(painter, label, 3, point.y(), background, align_left=True)
            y += spacing
        painter.restore()

    @staticmethod
    def _draw_coordinate_label(
        painter: QPainter,
        label: str,
        x: int,
        y: int,
        background: QColor,
        *,
        align_left: bool = False,
    ) -> None:
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(label)
        height = metrics.height()
        viewport = painter.viewport()
        if align_left:
            left = x + 2
            top = max(2, min(viewport.height() - height - 2, y - height // 2))
        else:
            left = max(2, min(viewport.width() - width - 2, x - width // 2))
            top = y - height - 2
        painter.fillRect(left - 2, top, width + 4, height, background)
        painter.drawText(left, top + metrics.ascent(), label)

    def set_mesh(self, mesh: FiniteVolumeMesh | None) -> None:
        self.mesh = mesh
        self.selected_patch_id = None
        self.solution = None
        self._cell_bounds = ()
        self._cell_centers = ()
        if mesh is not None:
            cell_bounds = []
            cell_centers = []
            for node_ids in mesh.cells:
                points = [mesh.points[node_id] for node_id in node_ids]
                cell_bounds.append(
                    (
                        min(point[0] for point in points),
                        min(point[1] for point in points),
                        max(point[0] for point in points),
                        max(point[1] for point in points),
                    )
                )
                twice_area = sum(
                    x0 * y1 - x1 * y0
                    for (x0, y0), (x1, y1) in zip(
                        points, points[1:] + points[:1]
                    )
                )
                if abs(twice_area) <= 1e-15:
                    center = (
                        sum(point[0] for point in points) / len(points),
                        sum(point[1] for point in points) / len(points),
                    )
                else:
                    x_sum = 0.0
                    y_sum = 0.0
                    for (x0, y0), (x1, y1) in zip(
                        points, points[1:] + points[:1]
                    ):
                        cross = x0 * y1 - x1 * y0
                        x_sum += (x0 + x1) * cross
                        y_sum += (y0 + y1) * cross
                    center = (
                        x_sum / (3.0 * twice_area),
                        y_sum / (3.0 * twice_area),
                    )
                cell_centers.append(center)
            self._cell_bounds = tuple(cell_bounds)
            self._cell_centers = tuple(cell_centers)
        self._smooth_cache_key = None
        self._smooth_cache = None
        self._draw_mesh(fit=True)

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in ("Navigate", "Box Zoom", "Cell Info"):
            raise ValueError(f"Unsupported canvas interaction mode: {mode}")
        self.interaction_mode = mode
        self._zoom_start = None
        self._zoom_rubber_band.hide()
        self.setCursor(
            Qt.CrossCursor if mode in ("Box Zoom", "Cell Info") else Qt.ArrowCursor
        )

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
        color_range: tuple[float, float] | None = None,
        smooth_colors: bool = False,
        show_colorbar: bool = False,
        colorbar_orientation: str = "Horizontal",
        colorbar_variable: str = "Density",
    ) -> tuple[float, ...]:
        values = self._solution_values(solution_variable)
        colors = color_values(values, color_map, color_scale, color_range)
        self.show_grid = show_grid
        self.show_mesh = show_mesh
        self.solution_variable = solution_variable
        self.color_map = color_map
        self.color_scale = color_scale
        self.color_range = color_range
        self.smooth_colors = smooth_colors
        self.show_colorbar = show_colorbar
        self.color_bar_overlay.set_values(
            values,
            color_map,
            color_scale,
            color_range,
            colorbar_orientation,
            colorbar_variable,
        )
        self.color_bar_overlay.setVisible(show_colorbar)
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
        colors = color_values(
            values, self.color_map, self.color_scale, self.color_range
        )
        use_smooth = self.smooth_colors and bool(values)
        if use_smooth:
            image, bounds = self._smooth_solution_image(values)
            image_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
            image_item.setPos(bounds.topLeft())
            image_item.setScale(max(bounds.width(), bounds.height()) / 1200)
            self.scene().addItem(image_item)
        else:
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
                item.setData(0, cell_id)
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

    def _smooth_solution_image(
        self, values: tuple[float, ...]
    ) -> tuple[QImage, QRectF]:
        assert self.mesh is not None
        cache_key = (
            id(self.mesh),
            values,
            self.color_map,
            self.color_scale,
            self.color_range,
        )
        if self._smooth_cache_key == cache_key and self._smooth_cache is not None:
            return self._smooth_cache

        points = self.mesh.points
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)
        longest_side = 1200
        image_width = max(2, round(longest_side * span_x / max(span_x, span_y)))
        image_height = max(2, round(longest_side * span_y / max(span_x, span_y)))
        image = QImage(image_width, image_height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        minimum, maximum = self.color_range or (min(values), max(values))
        scalar_values = (
            tuple(math.log10(value) for value in values)
            if self.color_scale == "Logarithmic"
            else values
        )
        if self.color_scale == "Logarithmic":
            minimum, maximum = math.log10(minimum), math.log10(maximum)
        scalar_span = maximum - minimum
        if not math.isfinite(scalar_span):
            divisor = max(abs(minimum), abs(maximum))
            minimum /= divisor
            maximum /= divisor
            scalar_values = tuple(value / divisor for value in scalar_values)
            scalar_span = maximum - minimum

        nodal_values: list[list[float]] = [[] for _ in points]
        for cell_id, cell in enumerate(self.mesh.cells):
            for node_id in cell:
                nodal_values[node_id].append(scalar_values[cell_id])
        node_values = [
            sum(incident_values) / len(incident_values)
            if incident_values
            else 0.0
            for incident_values in nodal_values
        ]

        def image_point(point: tuple[float, float]) -> tuple[float, float]:
            return (
                (point[0] - min_x) / span_x * (image_width - 1),
                (point[1] - min_y) / span_y * (image_height - 1),
            )

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)
        palette_colors = tuple(QColor(stop) for stop in COLOR_MAPS[self.color_map])
        palette_lut = tuple(
            _palette_color(index / 255, self.color_map) for index in range(256)
        )

        def color_for_value(value: float) -> QColor:
            position = (
                0.5
                if scalar_span == 0.0
                else max(0.0, min(1.0, (value - minimum) / scalar_span))
            )
            return palette_lut[round(position * 255)]

        for cell in self.mesh.cells:
            mapped = tuple(image_point(points[node_id]) for node_id in cell)
            vertex_values = tuple(node_values[node_id] for node_id in cell)
            center_x = sum(point[0] for point in mapped) / len(mapped)
            center_y = sum(point[1] for point in mapped) / len(mapped)
            center_value = sum(vertex_values) / len(vertex_values)
            sxx = sxy = syy = sxv = syv = 0.0
            for (x, y), value in zip(mapped, vertex_values):
                dx, dy, dv = x - center_x, y - center_y, value - center_value
                sxx += dx * dx
                sxy += dx * dy
                syy += dy * dy
                sxv += dx * dv
                syv += dy * dv
            determinant = sxx * syy - sxy * sxy
            polygon = QPolygonF([QPointF(x, y) for x, y in mapped])
            scalar_low = min(vertex_values)
            scalar_high = max(vertex_values)
            if determinant < 1e-12 or scalar_low == scalar_high:
                color_position = (
                    0.5
                    if scalar_span == 0.0
                    else (center_value - minimum) / scalar_span
                )
                painter.setBrush(
                    palette_lut[
                        round(max(0.0, min(1.0, color_position)) * 255)
                    ]
                )
            else:
                slope_x = (sxv * syy - syv * sxy) / determinant
                slope_y = (syv * sxx - sxv * sxy) / determinant
                gradient_norm = slope_x * slope_x + slope_y * slope_y
                if gradient_norm < 1e-20:
                    color_position = (
                        0.5
                        if scalar_span == 0.0
                        else (center_value - minimum) / scalar_span
                    )
                    painter.setBrush(_palette_color(color_position, self.color_map))
                    painter.drawPolygon(polygon)
                    continue
                constant = center_value - slope_x * center_x - slope_y * center_y
                start = QPointF(
                    slope_x * (scalar_low - constant) / gradient_norm,
                    slope_y * (scalar_low - constant) / gradient_norm,
                )
                end = QPointF(
                    slope_x * (scalar_high - constant) / gradient_norm,
                    slope_y * (scalar_high - constant) / gradient_norm,
                )
                gradient = QLinearGradient(start, end)
                gradient.setColorAt(0.0, color_for_value(scalar_low))
                gradient.setColorAt(1.0, color_for_value(scalar_high))
                for stop_index, color in enumerate(palette_colors[1:-1], 1):
                    stop_value = minimum + scalar_span * (
                        stop_index / (len(palette_colors) - 1)
                    )
                    if scalar_low < stop_value < scalar_high:
                        gradient.setColorAt(
                            (stop_value - scalar_low) / (scalar_high - scalar_low),
                            color,
                        )
                painter.setBrush(gradient)
            painter.drawPolygon(polygon)

        painter.end()
        result = image, QRectF(min_x, min_y, span_x, span_y)
        self._smooth_cache_key = cache_key
        self._smooth_cache = result
        return result

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.color_bar_overlay.move_to(self.color_bar_overlay.pos())

    def reset_zoom(self) -> None:
        if self.mesh is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def _cell_at(self, scene_position: QPointF) -> int | None:
        if self.mesh is None:
            return None
        for item in self.scene().items(scene_position):
            cell_id = item.data(0)
            if isinstance(item, QGraphicsPolygonItem) and cell_id is not None:
                return int(cell_id)
        x, y = scene_position.x(), scene_position.y()
        for cell_id, bounds in enumerate(self._cell_bounds):
            min_x, min_y, max_x, max_y = bounds
            if min_x <= x <= max_x and min_y <= y <= max_y:
                polygon = QPolygonF(
                    [
                        QPointF(*self.mesh.points[node_id])
                        for node_id in self.mesh.cells[cell_id]
                    ]
                )
                if polygon.containsPoint(scene_position, Qt.OddEvenFill):
                    return cell_id
        return None

    def _emit_cell_info(self, viewport_position: QPoint) -> None:
        if self.mesh is None:
            return
        cell_id = self._cell_at(self.mapToScene(viewport_position))
        if cell_id is None:
            return
        center_x, center_y = self._cell_centers[cell_id]
        value = (
            getattr(self.solution[cell_id], SOLUTION_VARIABLES[self.solution_variable])
            if self.solution is not None
            else None
        )
        self.cell_selected.emit(
            cell_id, center_x, center_y, self.solution_variable, value
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.interaction_mode == "Box Zoom":
            self._zoom_start = event.pos()
            self._zoom_rubber_band.setGeometry(QRect(self._zoom_start, self._zoom_start))
            self._zoom_rubber_band.show()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.interaction_mode == "Cell Info":
            self._emit_cell_info(event.pos())
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
        if self._zoom_start is not None and event.buttons() & Qt.LeftButton:
            self._zoom_rubber_band.setGeometry(
                QRect(self._zoom_start, event.pos()).normalized()
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
        if (
            event.button() == Qt.LeftButton
            and self._zoom_start is not None
            and self.interaction_mode == "Box Zoom"
        ):
            selection = QRect(self._zoom_start, event.pos()).normalized()
            self._zoom_rubber_band.hide()
            self._zoom_start = None
            if selection.width() >= 4 and selection.height() >= 4:
                scene_bounds = self.mapToScene(selection).boundingRect()
                if scene_bounds.width() > 0.0 and scene_bounds.height() > 0.0:
                    self.fitInView(scene_bounds, Qt.KeepAspectRatio)
            event.accept()
            return
        super().mouseReleaseEvent(event)
