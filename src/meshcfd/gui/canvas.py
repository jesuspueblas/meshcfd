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
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QRubberBand,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from meshcfd.core.flow_visualization import (
    contour_segments, interpolate_to_regular_grid, streamline_polylines, vector_samples,
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
    "Mach number": "mach_number",
    "Total pressure": "total_pressure",
    "Total temperature": "total_temperature",
    "Total density": "total_density",
    "y+ (wall distance)": "y_plus",
    "Wall distance": "wall_distance",
    "Regularized log(omega)": "log_regularized_omega",
}


def available_solution_variables(
    turbulence_model: str, omega_formulation: str, viscosity_model: str
) -> tuple[str, ...]:
    variables = [
        name
        for name in SOLUTION_VARIABLES
        if name not in ("Wall distance", "Regularized log(omega)")
    ]
    if turbulence_model == "Menter k-omega SST":
        variables.append("Wall distance")
        if (
            omega_formulation == "Regularized k-log-omega"
            and viscosity_model != "Inviscid"
        ):
            variables.append("Regularized log(omega)")
    return tuple(variables)


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
            label_width = min(54, max(20, self.width() // 3))
            top_margin = min(28, max(12, self.height() // 5))
            bar_width = max(1, self.width() - label_width - 20)
            bar_height = max(1, self.height() - top_margin - 12)
            bar = QRect(8, top_margin, bar_width, bar_height)
            gradient = QLinearGradient(0, bar.bottom(), 0, bar.top())
        else:
            horizontal_margin = min(28, max(8, self.width() // 8))
            top_margin = min(24, max(10, self.height() // 3))
            bottom_margin = min(20, max(6, self.height() // 4))
            bar = QRect(
                horizontal_margin,
                top_margin,
                max(1, self.width() - 2 * horizontal_margin),
                max(1, self.height() - top_margin - bottom_margin),
            )
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
                    painter.drawLine(
                        bar.right(), position, min(self.width() - 1, bar.right() + 3), position
                    )
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
            self.overlay.remember_size_fraction()
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
        self._position = QPoint(16, 16)
        self.orientation = "Horizontal"
        self.setObjectName("color_bar_overlay")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(1, 1)
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
        self._size_fraction: tuple[float, float] | None = None
        self.move_to(self._position)
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
        orientation_changed = orientation != self.orientation
        if orientation_changed:
            self.orientation = orientation
            self._size_fraction = None
        self.resize_to_viewport()
        self.move_to(self._position)

    def resize_to_viewport(self) -> None:
        bounds = self.parentWidget().size()
        if self._size_fraction is None:
            if bounds.width() <= 0 or bounds.height() <= 0:
                return
            default_size = (
                QSize(150, 250)
                if self.orientation == "Vertical"
                else QSize(310, 72)
            )
            if (
                bounds.width() < default_size.width()
                or bounds.height() < default_size.height()
            ):
                self.resize(
                    min(bounds.width(), default_size.width()),
                    min(bounds.height(), default_size.height()),
                )
                return
            self._size_fraction = (
                min(0.9, default_size.width() / bounds.width()),
                min(0.9, default_size.height() / bounds.height()),
            )
        width_fraction, height_fraction = self._size_fraction
        width_fraction = min(width_fraction, 0.95)
        height_fraction = min(height_fraction, 0.95)
        width = max(1, min(bounds.width(), round(bounds.width() * width_fraction)))
        height = max(1, min(bounds.height(), round(bounds.height() * height_fraction)))
        self.resize(width, height)

    def remember_size_fraction(self) -> None:
        bounds = self.parentWidget().size()
        if bounds.width() > 0 and bounds.height() > 0:
            self._size_fraction = (
                self.width() / bounds.width(),
                self.height() / bounds.height(),
            )

    def move_to(self, position: QPoint) -> None:
        bounds = self.parentWidget().rect()
        x = max(0, min(position.x(), max(0, bounds.width() - self.width())))
        y = max(0, min(position.y(), max(0, bounds.height() - self.height())))
        self._position = QPoint(x, y)
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
        self._solution_cut: tuple[float, float, float] | None = None
        self.solution: tuple[InitialState, ...] | None = None
        self.derived_solution_fields: dict[str, tuple[float, ...]] = {}
        self.show_grid = True
        self.show_mesh = True
        self.solution_variable = "Density"
        self.color_map = "Viridis"
        self.color_scale = "Linear"
        self.color_range: tuple[float, float] | None = None
        self.smooth_colors = False
        self.show_colorbar = False
        self.show_contours = False
        self.contour_variable = "Pressure"
        self.contour_levels = 15
        self.show_vectors = False
        self.show_streamlines = False
        self.overlay_density = 2
        self._overlay_cache_key = None
        self._overlay_grid = None
        self.interaction_mode = "Navigate"
        self._panning = False
        self._last_mouse_pos = None
        self._zoom_start: QPoint | None = None
        self._zoom_rubber_band = QRubberBand(
            QRubberBand.Rectangle, self.viewport()
        )
        self._cell_bounds: tuple[tuple[float, float, float, float], ...] = ()
        self._cell_centers: tuple[tuple[float, float], ...] = ()
        self._content_scene_rect = QRectF()
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
        self._solution_cut = None
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

    def set_solution(
        self,
        solution: tuple[InitialState, ...] | None,
        derived_fields: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        self.solution = solution
        self.derived_solution_fields = derived_fields or {}

    def set_solution_cut(
        self, point_x: float, point_y: float, angle_degrees: float
    ) -> None:
        if not all(math.isfinite(value) for value in (point_x, point_y, angle_degrees)):
            raise ValueError("Cut point and angle must be finite.")
        self._solution_cut = (point_x, point_y, angle_degrees)
        self._draw_mesh()
        self.viewport().update()

    def clear_solution_cut(self) -> None:
        if self._solution_cut is None:
            return
        self._solution_cut = None
        self._draw_mesh()
        self.viewport().update()

    def _draw_solution_cut(self) -> None:
        if self.mesh is None or self._solution_cut is None or not self.mesh.points:
            return
        point_x, point_y, angle_degrees = self._solution_cut
        direction_x = math.cos(math.radians(angle_degrees))
        direction_y = math.sin(math.radians(angle_degrees))
        bounds = (
            min(x for x, _ in self.mesh.points),
            min(y for _, y in self.mesh.points),
            max(x for x, _ in self.mesh.points),
            max(y for _, y in self.mesh.points),
        )
        scale = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0)
        tolerance = scale * 1e-12
        lower = -math.inf
        upper = math.inf
        for coordinate, direction, minimum, maximum in (
            (point_x, direction_x, bounds[0], bounds[2]),
            (point_y, direction_y, bounds[1], bounds[3]),
        ):
            if abs(direction) <= 1e-15:
                if coordinate < minimum - tolerance or coordinate > maximum + tolerance:
                    return
                continue
            first = (minimum - coordinate) / direction
            second = (maximum - coordinate) / direction
            lower = max(lower, min(first, second))
            upper = min(upper, max(first, second))
        if not math.isfinite(lower) or not math.isfinite(upper) or upper - lower <= tolerance:
            return

        start_x = point_x + lower * direction_x
        start_y = point_y + lower * direction_y
        end_x = point_x + upper * direction_x
        end_y = point_y + upper * direction_y
        underlay = QGraphicsLineItem(start_x, start_y, end_x, end_y)
        underlay_pen = QPen(QColor("#FFFFFF"), 5.0)
        underlay_pen.setCosmetic(True)
        underlay.setPen(underlay_pen)
        underlay.setZValue(1000)
        self.scene().addItem(underlay)
        line = QGraphicsLineItem(start_x, start_y, end_x, end_y)
        line_pen = QPen(QColor("#F43F5E"), 2.5)
        line_pen.setCosmetic(True)
        line.setPen(line_pen)
        line.setData(0, "solution_cut")
        line.setZValue(1001)
        self.scene().addItem(line)

        if (
            bounds[0] - tolerance <= point_x <= bounds[2] + tolerance
            and bounds[1] - tolerance <= point_y <= bounds[3] + tolerance
        ):
            radius = scale * 0.006
            marker = QGraphicsEllipseItem(
                point_x - radius, point_y - radius, 2.0 * radius, 2.0 * radius
            )
            marker.setPen(QPen(QColor("#FFFFFF"), 3.0))
            marker.setBrush(QColor("#F43F5E"))
            marker.setData(0, "solution_cut_origin")
            marker.setZValue(1002)
            self.scene().addItem(marker)

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
        show_contours: bool = False,
        contour_variable: str = "Pressure",
        contour_levels: int = 15,
        show_vectors: bool = False,
        show_streamlines: bool = False,
        overlay_density: int = 2,
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
        self.show_contours = show_contours
        self.contour_variable = contour_variable
        self.contour_levels = max(2, int(contour_levels))
        self.show_vectors = show_vectors
        self.show_streamlines = show_streamlines
        self.overlay_density = max(1, min(4, int(overlay_density)))
        self.color_bar_overlay.set_values(
            values,
            color_map,
            color_scale,
            color_range,
            colorbar_orientation,
            colorbar_variable,
        )
        self.color_bar_overlay.setVisible(show_colorbar and bool(values))
        self._draw_mesh()
        self.viewport().update()
        return values if colors else ()

    def _solution_values(self, variable: str) -> tuple[float, ...]:
        if self.mesh is None or self.solution is None:
            return ()
        if len(self.solution) != len(self.mesh.cells):
            raise ValueError("Initial solution does not match the loaded mesh.")
        field = SOLUTION_VARIABLES[variable]
        if field in self.derived_solution_fields:
            values = self.derived_solution_fields[field]
            if len(values) != len(self.mesh.cells):
                raise ValueError("Derived solution field does not match the loaded mesh.")
            return values
        return tuple(getattr(state, field) for state in self.solution)

    def select_patch(self, patch_id: int | None) -> None:
        if self.selected_patch_id == patch_id:
            return
        self.selected_patch_id = patch_id
        self._draw_mesh()

    def _draw_mesh(self, fit: bool = False) -> None:
        self.scene().clear()
        if self.mesh is None:
            self._content_scene_rect = QRectF()
            self.scene().setSceneRect(QRectF(-1.0, -1.0, 2.0, 2.0))
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
                    pen = QPen(QColor("#000000"), 0.7)
                    pen.setCosmetic(True)
                    item.setPen(pen)
                else:
                    item.setPen(QPen(Qt.NoPen))
                fill = colors[cell_id] if colors else QColor("#F1F5F9")
                item.setBrush(QBrush(fill))
                item.setData(0, cell_id)
                self.scene().addItem(item)

        if not self.show_mesh:
            self._draw_solution_cut()
        else:
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
                    color = QColor("#000000")
                    width = 0.7
                if not selected:
                    color = QColor("#000000")
                pen = QPen(color, width)
                pen.setCosmetic(True)
                item = QGraphicsLineItem(x0, y0, x1, y1)
                item.setPen(pen)
                self.scene().addItem(item)

            self._draw_solution_cut()

        # Draw flow overlays after cells, mesh lines and the solution cut so the
        # overlay artists remain visible and are recreated on every refresh.
        self._draw_flow_overlays()

        if fit:
            self._set_content_scene_rect(self._mesh_content_bounds(), fit=True)

    def _draw_flow_overlays(self) -> None:
        if self.mesh is None or self.solution is None or not (self.show_contours or self.show_vectors or self.show_streamlines):
            return
        scalar = self._solution_values(self.contour_variable)
        velocity_x = self._solution_values("Velocity X")
        velocity_y = self._solution_values("Velocity Y")
        key = (id(self.mesh), id(self.solution), self.contour_variable, self.overlay_density)
        if key != self._overlay_cache_key:
            resolution = (50 + 20 * self.overlay_density, 35 + 12 * self.overlay_density)
            self._overlay_grid = interpolate_to_regular_grid(self.mesh, scalar, velocity_x, velocity_y, resolution)
            self._overlay_cache_key = key
        grid = self._overlay_grid
        if self.show_contours:
            pen = QPen(QColor(20, 20, 20, 190), 0)
            for start, end, _level in contour_segments(grid, self.contour_levels):
                item = self.scene().addLine(start[0], start[1], end[0], end[1], pen)
                item.setData(0, "flow_overlay")
                item.setZValue(20)
        if self.show_vectors:
            self._draw_velocity_vectors(grid)
        if self.show_streamlines:
            pen = QPen(QColor("#F8FAFC"), 0)
            for line in streamline_polylines(grid, self.overlay_density):
                path = QPainterPath(QPointF(*line[0]))
                for point in line[1:]:
                    path.lineTo(QPointF(*point))
                item = QGraphicsPathItem(path)
                item.setPen(pen)
                item.setData(0, "flow_overlay")
                item.setZValue(22)
                self.scene().addItem(item)

    def _draw_velocity_vectors(self, grid) -> None:
        samples = vector_samples(grid, self.overlay_density)
        if not samples:
            return
        maximum = max(math.hypot(u, v) for _x, _y, u, v in samples)
        length_scale = 0.75 * min(float(grid.x[1] - grid.x[0]), float(grid.y[1] - grid.y[0]))
        pen = QPen(QColor("#111827"), 0)
        for x, y, u, v in samples:
            speed = math.hypot(u, v)
            if speed <= 1e-14 or maximum <= 1e-14:
                continue
            length = length_scale * (0.25 + 0.75 * speed / maximum)
            dx, dy = length * u / speed, length * v / speed
            end_x, end_y = x + dx, y + dy
            angle, head = math.atan2(dy, dx), 0.28 * length
            segments = (((x, y), (end_x, end_y)), ((end_x, end_y), (end_x - head * math.cos(angle - 0.55), end_y - head * math.sin(angle - 0.55))), ((end_x, end_y), (end_x - head * math.cos(angle + 0.55), end_y - head * math.sin(angle + 0.55))))
            for first, second in segments:
                item = self.scene().addLine(first[0], first[1], second[0], second[1], pen)
                item.setData(0, "flow_overlay")
                item.setZValue(25)

    def _mesh_content_bounds(self) -> QRectF:
        assert self.mesh is not None
        if not self.mesh.points:
            return QRectF(-0.5, -0.5, 1.0, 1.0)
        min_x = min(point[0] for point in self.mesh.points)
        min_y = min(point[1] for point in self.mesh.points)
        max_x = max(point[0] for point in self.mesh.points)
        max_y = max(point[1] for point in self.mesh.points)
        bounds = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
        if bounds.width() < 1e-9:
            bounds.adjust(-0.5, 0.0, 0.5, 0.0)
        if bounds.height() < 1e-9:
            bounds.adjust(0.0, -0.5, 0.0, 0.5)
        margin = max(1.0, max(bounds.width(), bounds.height()) * 0.04)
        bounds.adjust(-margin, -margin, margin, margin)
        return bounds

    def _set_content_scene_rect(self, bounds: QRectF, fit: bool) -> None:
        self._content_scene_rect = QRectF(bounds)
        self.scene().setSceneRect(bounds)
        if fit:
            self.reset_zoom()
        self._expand_scene_rect_for_panning()

    def _expand_scene_rect_for_panning(self) -> None:
        if self._content_scene_rect.isEmpty():
            return
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        center = visible.center()
        width = max(visible.width() * 4.0, self._content_scene_rect.width() * 2.0)
        height = max(visible.height() * 4.0, self._content_scene_rect.height() * 2.0)
        bounds = QRectF(
            center.x() - width / 2.0,
            center.y() - height / 2.0,
            width,
            height,
        )
        self.scene().setSceneRect(bounds.united(self._content_scene_rect))
        self.centerOn(center)

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
        self._expand_scene_rect_for_panning()
        self.color_bar_overlay.resize_to_viewport()
        self.color_bar_overlay.move_to(self.color_bar_overlay._position)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if hasattr(self, "color_bar_overlay"):
            self.color_bar_overlay.move_to(self.color_bar_overlay._position)

    def reset_zoom(self) -> None:
        if self.mesh is not None and not self._content_scene_rect.isEmpty():
            self.fitInView(self._content_scene_rect, Qt.KeepAspectRatio)
            self._expand_scene_rect_for_panning()
            self.centerOn(self._content_scene_rect.center())

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
        values = self._solution_values(self.solution_variable)
        value = values[cell_id] if values else None
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
