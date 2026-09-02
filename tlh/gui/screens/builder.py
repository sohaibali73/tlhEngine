"""TLH model builder: drag blocks from the palette onto the canvas, order them left to right, edit their
parameters, run the pipeline (screen -> construct -> transition -> harvest -> save/export), save/load/share as JSON
or hand it to YANG."""
from __future__ import annotations

import json
import uuid

from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QDrag, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...optim.pipeline import EXAMPLES, NODE_TYPES, Node, Pipeline, new_node, ordered, validate
from ...services.pipeline_service import PipelineService
from .. import theme
from ..widgets import FrameTable, KpiCard, button, header, pct, vbox
from ..workers import run_task

MIME = "application/x-tlh-node"
NODE_W, NODE_H = 176, 64


def _summary(n: Node) -> str:
    p = n.params
    t = n.type
    if t == "universe":
        return f"{p.get('source')} {p.get('name') or ''}".strip()
    if t == "filter":
        bits = []
        if p.get("min_mktcap_musd"):
            bits.append(f"mcap≥{p['min_mktcap_musd']:,.0f}M")
        if p.get("sectors_include"):
            bits.append(f"in:{p['sectors_include'][:18]}")
        if p.get("sectors_exclude"):
            bits.append(f"ex:{p['sectors_exclude'][:18]}")
        return ", ".join(bits) or "no filters"
    if t == "rank":
        return f"{p.get('signal_weights', '')[:22]} → top {p.get('top_n')}"
    if t == "benchmark":
        return str(p.get("name"))
    if t == "construct":
        return f"{p.get('strategy')} · n≤{p.get('n_max')} · w≤{float(p.get('max_weight', 0)):.0%}"
    if t == "transition":
        return f"gain≤{float(p.get('gain_budget', 0)):.2%} · turn≤{float(p.get('turnover_max', 0)):.0%}"
    if t == "harvest":
        return f"{p.get('mode')} · TE {float(p.get('te_budget', 0)):.1%}"
    if t == "output":
        return p.get("basket_name") or "(no basket name)"
    return ""


class NodeItem(QGraphicsRectItem):
    def __init__(self, node: Node, canvas):
        super().__init__(0, 0, NODE_W, NODE_H)
        self.node = node
        self.canvas = canvas
        self.setPos(node.x, node.y)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        color = QColor(NODE_TYPES[node.type]["color"])
        self.setBrush(QBrush(QColor(theme.BG3)))
        self.setPen(QPen(color, 2))
        self.title = QGraphicsSimpleTextItem(NODE_TYPES[node.type]["title"], self)
        self.title.setBrush(QBrush(color))
        f = QFont("Segoe UI", 10)
        f.setBold(True)
        self.title.setFont(f)
        self.title.setPos(10, 8)
        self.sub = QGraphicsSimpleTextItem("", self)
        self.sub.setBrush(QBrush(QColor(theme.TEXT)))
        self.sub.setFont(QFont("Segoe UI", 8))
        self.sub.setPos(10, 34)
        self.refresh()

    def refresh(self) -> None:
        s = _summary(self.node)
        self.sub.setText(s if len(s) <= 30 else s[:29] + "…")
        self.setToolTip(NODE_TYPES[self.node.type]["help"] + "\n\n" + json.dumps(self.node.params, indent=1, default=str)[:600])

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.x, self.node.y = self.pos().x(), self.pos().y()
            self.canvas.layout_changed()
        if change == QGraphicsItem.ItemSelectedHasChanged and value:
            self.canvas.node_selected.emit(self.node.id)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        if self.isSelected():
            painter.setPen(QPen(QColor(theme.ACCENT2), 3))
        else:
            painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawRoundedRect(self.rect(), 8, 8)


class Canvas(QGraphicsView):
    node_selected = Signal(str)
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.scene_.setSceneRect(QRectF(-50, -50, 2400, 900))
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor(theme.BG)))
        self.setAcceptDrops(True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.items_: dict[str, NodeItem] = {}
        self.edges: list[QGraphicsPathItem] = []
        self.pipeline = Pipeline()
        self.hint = self.scene_.addSimpleText("Drag blocks from the palette. Left-to-right order = execution order. Delete key removes.")
        self.hint.setBrush(QBrush(QColor(theme.MUTED)))
        self.hint.setPos(20, 20)

    # ------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        if not ev.mimeData().hasFormat(MIME):
            return
        node_type = bytes(ev.mimeData().data(MIME)).decode()
        pos = self.mapToScene(ev.position().toPoint())
        self.add_node(node_type, pos.x() - NODE_W / 2, pos.y() - NODE_H / 2)
        ev.acceptProposedAction()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for it in list(self.scene_.selectedItems()):
                if isinstance(it, NodeItem):
                    self.remove_node(it.node.id)
            return
        super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
            self.scale(f, f)
        else:
            super().wheelEvent(ev)

    # ------------------------------------------------------------------ model
    def add_node(self, node_type: str, x: float | None = None, y: float = 120.0, **params) -> Node:
        if x is None:
            x = max([n.x for n in self.pipeline.nodes], default=-220) + 220
        node = new_node(node_type, uuid.uuid4().hex[:8], x, y, **params)
        self.pipeline.nodes.append(node)
        item = NodeItem(node, self)
        self.scene_.addItem(item)
        self.items_[node.id] = item
        self.hint.setVisible(False)
        self.layout_changed()
        item.setSelected(True)
        return node

    def remove_node(self, node_id: str) -> None:
        item = self.items_.pop(node_id, None)
        if item:
            self.scene_.removeItem(item)
        self.pipeline.nodes = [n for n in self.pipeline.nodes if n.id != node_id]
        self.hint.setVisible(not self.pipeline.nodes)
        self.layout_changed()

    def load(self, p: Pipeline) -> None:
        for it in list(self.items_.values()):
            self.scene_.removeItem(it)
        self.items_.clear()
        self.pipeline = Pipeline(p.name, [], p.description)
        for n in p.nodes:
            node = Node(n.id, n.type, dict(n.params), n.x, n.y)
            self.pipeline.nodes.append(node)
            item = NodeItem(node, self)
            self.scene_.addItem(item)
            self.items_[node.id] = item
        self.hint.setVisible(not self.pipeline.nodes)
        self.layout_changed()

    def auto_layout(self) -> None:
        for i, n in enumerate(ordered(self.pipeline)):
            n.x, n.y = i * 220.0, 120.0
            self.items_[n.id].setPos(n.x, n.y)
        self.layout_changed()

    def layout_changed(self) -> None:
        for e in self.edges:
            self.scene_.removeItem(e)
        self.edges.clear()
        seq = ordered(self.pipeline)
        for a, b in zip(seq, seq[1:], strict=False):
            ia, ib = self.items_.get(a.id), self.items_.get(b.id)
            if not ia or not ib:
                continue
            p1 = ia.pos() + QPointF(NODE_W, NODE_H / 2)
            p2 = ib.pos() + QPointF(0, NODE_H / 2)
            path = QPainterPath(p1)
            dx = max((p2.x() - p1.x()) / 2, 40)
            path.cubicTo(p1 + QPointF(dx, 0), p2 - QPointF(dx, 0), p2)
            edge = QGraphicsPathItem(path)
            edge.setPen(QPen(QColor(theme.BORDER), 2))
            edge.setZValue(-1)
            self.scene_.addItem(edge)
            self.edges.append(edge)
            # arrow head
            head = QPainterPath(p2)
            head.lineTo(p2 + QPointF(-10, -5))
            head.lineTo(p2 + QPointF(-10, 5))
            head.closeSubpath()
            h = QGraphicsPathItem(head)
            h.setBrush(QBrush(QColor(theme.BORDER)))
            h.setPen(QPen(Qt.NoPen))
            h.setZValue(-1)
            self.scene_.addItem(h)
            self.edges.append(h)
        self.changed.emit()

    def refresh_node(self, node_id: str) -> None:
        it = self.items_.get(node_id)
        if it:
            it.refresh()
        self.layout_changed()


class Palette(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        for t, spec in NODE_TYPES.items():
            it = QListWidgetItem(f"■ {spec['title']}   ({spec['category']})")
            it.setData(Qt.UserRole, t)
            it.setForeground(QColor(spec["color"]))
            it.setToolTip(spec["help"])
            self.addItem(it)

    def startDrag(self, actions):
        it = self.currentItem()
        if not it:
            return
        md = QMimeData()
        md.setData(MIME, it.data(Qt.UserRole).encode())
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.exec(Qt.CopyAction)


class PropertyPanel(QScrollArea):
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.setWidget(self.body)
        self.node: Node | None = None
        self.title = QLabel("Select a block")
        self.title.setProperty("h1", True)
        self.form.addRow(self.title)

    def show_node(self, node: Node | None) -> None:
        while self.form.rowCount() > 0:
            self.form.removeRow(0)
        self.node = node
        if node is None:
            self.form.addRow(QLabel("Select a block to edit its parameters."))
            return
        spec = NODE_TYPES[node.type]
        t = QLabel(spec["title"])
        t.setProperty("h1", True)
        self.form.addRow(t)
        h = QLabel(spec["help"])
        h.setWordWrap(True)
        h.setProperty("muted", True)
        self.form.addRow(h)
        for p in spec["params"]:
            w = self._widget(p, node.params.get(p["name"], p["default"]))
            if p.get("help"):
                w.setToolTip(p["help"])
            self.form.addRow(p["label"], w)

    def _widget(self, p: dict, value):
        name, kind = p["name"], p["type"]
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda v, n=name: self._set(n, bool(v)))
        elif kind == "int":
            w = QSpinBox()
            w.setRange(0, 100000)
            w.setValue(int(value or 0))
            w.valueChanged.connect(lambda v, n=name: self._set(n, int(v)))
        elif kind == "float":
            w = QDoubleSpinBox()
            w.setRange(-1e9, 1e9)
            w.setDecimals(4)
            w.setValue(float(value or 0))
            w.valueChanged.connect(lambda v, n=name: self._set(n, float(v)))
        elif kind == "choice":
            w = QComboBox()
            w.addItems(p["choices"])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(lambda v, n=name: self._set(n, v))
        else:
            w = QLineEdit(str(value or ""))
            w.editingFinished.connect(lambda n=name, ww=w: self._set(n, ww.text()))
        return w

    def _set(self, name: str, value) -> None:
        if self.node is not None:
            self.node.params[name] = value
            self.changed.emit(self.node.id)


class BuilderScreen(QWidget):
    data_changed = Signal()

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.ctx = app.ctx
        self.svc = PipelineService(self.ctx)
        self._build()
        self.canvas.load(EXAMPLES["Quality core + harvest"])
        self.name.setText(self.canvas.pipeline.name)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        top.addWidget(header("TLH model builder", "Drag blocks, order them left to right, tune parameters, run. Universe → Filter/Rank → Benchmark → Construction → Transition → Harvest → Save/export."))
        top.addStretch(1)
        root.addLayout(top)
        bar = QHBoxLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("model name")
        self.name.setMinimumWidth(220)
        self.name.editingFinished.connect(lambda: setattr(self.canvas.pipeline, "name", self.name.text().strip() or "Untitled TLH model"))
        self.examples = QComboBox()
        self.examples.addItem("Examples…")
        self.examples.addItems(list(EXAMPLES))
        self.examples.currentIndexChanged.connect(self._example)
        self.saved = QComboBox()
        self.saved.setMinimumWidth(200)
        self.saved.addItem("Saved models…")
        self.saved.currentIndexChanged.connect(self._load_saved)
        bar.addWidget(QLabel("Name"))
        bar.addWidget(self.name)
        bar.addWidget(self.examples)
        bar.addWidget(self.saved)
        bar.addWidget(button("New", self._new))
        bar.addWidget(button("Save", self._save))
        bar.addWidget(button("Delete saved", self._delete, danger=True))
        bar.addWidget(button("Auto-layout", lambda: self.canvas.auto_layout()))
        bar.addWidget(button("Import JSON", self._import))
        bar.addWidget(button("Export JSON", self._export))
        bar.addStretch(1)
        bar.addWidget(button("Ask YANG to design…", self._ask_yang, tooltip="Describe the model in words; YANG drafts the blocks"))
        self.run_btn = button("Run model", self.run, primary=True)
        bar.addWidget(self.run_btn)
        root.addLayout(bar)
        self.valid = QLabel("")
        self.valid.setWordWrap(True)
        root.addWidget(self.valid)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        self.palette = Palette()
        self.palette.itemDoubleClicked.connect(lambda it: self.canvas.add_node(it.data(Qt.UserRole)))
        pal = vbox(header("Blocks", "drag onto canvas · double-click appends"), self.palette)
        split.addWidget(pal)
        self.canvas = Canvas()
        self.canvas.node_selected.connect(self._select)
        self.canvas.changed.connect(self._validate)
        mid = QSplitter(Qt.Vertical)
        mid.addWidget(self.canvas)
        self.tabs = QTabWidget()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Run log")
        self.weights = FrameTable(["symbol", "weight"], pct_cols={"weight"})
        self.tabs.addTab(self.weights, "Result weights")
        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.json_view, "JSON")
        k = QHBoxLayout()
        self.k_uni = KpiCard("Universe after screens")
        self.k_n = KpiCard("Names")
        self.k_te = KpiCard("TE vs benchmark")
        self.k_run = KpiCard("Harvest run")
        for c in (self.k_uni, self.k_n, self.k_te, self.k_run):
            k.addWidget(c)
        kw = QWidget()
        kw.setLayout(k)
        mid.addWidget(vbox(kw, self.tabs))
        mid.setSizes([420, 320])
        split.addWidget(mid)
        self.props = PropertyPanel()
        self.props.changed.connect(self._prop_changed)
        split.addWidget(self.props)
        split.setSizes([200, 1000, 340])
        self._validate()

    # ------------------------------------------------------------------ editing
    def _select(self, node_id: str) -> None:
        node = next((n for n in self.canvas.pipeline.nodes if n.id == node_id), None)
        self.props.show_node(node)

    def _prop_changed(self, node_id: str) -> None:
        self.canvas.refresh_node(node_id)

    def _validate(self) -> None:
        errs = validate(self.canvas.pipeline)
        if errs:
            self.valid.setText("⚠ " + " · ".join(errs))
            self.valid.setStyleSheet(f"color: {theme.AMBER}")
        else:
            seq = " → ".join(NODE_TYPES[n.type]["title"] for n in ordered(self.canvas.pipeline))
            self.valid.setText("✓ " + seq)
            self.valid.setStyleSheet(f"color: {theme.GREEN}")
        self.json_view.setPlainText(self.canvas.pipeline.to_json())

    def _example(self, i: int) -> None:
        if i <= 0:
            return
        p = EXAMPLES[self.examples.currentText()]
        self.canvas.load(p)
        self.name.setText(p.name)
        self.examples.setCurrentIndex(0)
        self.props.show_node(None)

    def _new(self) -> None:
        self.canvas.load(Pipeline("Untitled TLH model"))
        self.name.setText("Untitled TLH model")
        self.props.show_node(None)

    def _save(self) -> None:
        self.canvas.pipeline.name = self.name.text().strip() or "Untitled TLH model"
        self.svc.save(self.canvas.pipeline)
        self.app.status(f"Saved TLH model '{self.canvas.pipeline.name}'.")
        self.refresh()
        self.data_changed.emit()

    def _delete(self) -> None:
        nm = self.name.text().strip()
        if nm and self.ctx.pipelines.get(nm) and QMessageBox.question(self, "Delete", f"Delete saved model '{nm}'?") == QMessageBox.Yes:
            self.svc.delete(nm)
            self.refresh()

    def _load_saved(self, i: int) -> None:
        if i <= 0:
            return
        p = self.svc.load(self.saved.currentText())
        if p:
            self.canvas.load(p)
            self.name.setText(p.name)
            self.props.show_node(None)
        self.saved.setCurrentIndex(0)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import TLH model JSON", "", "JSON (*.json)")
        if path:
            with open(path, encoding="utf-8") as f:
                p = Pipeline.from_json(f.read())
            self.canvas.load(p)
            self.name.setText(p.name)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export TLH model JSON", f"{self.name.text() or 'model'}.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.canvas.pipeline.to_json())

    def _ask_yang(self) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Ask YANG to design a TLH model",
                                                 "Describe the model in words. YANG will draft the blocks as a saved pipeline you can load here.",
                                                 "A 40-name quality/low-vol core from the S&P 500 excluding energy, min-variance construction with a 3% sector band, "
                                                 "then a tax-aware transition with a 0.5% gain budget and a harvest toward it. Save it as 'YANG core'.")
        if ok and text.strip():
            self.app.show_quick()
            self.app.quick.input.setText(f"Design a TLH model pipeline: {text.strip()} Use save_pipeline with valid block JSON (see pipeline_schema), then run_pipeline and report.")

    # ------------------------------------------------------------------ run
    def refresh(self) -> None:
        df = self.svc.list()
        self.saved.blockSignals(True)
        self.saved.clear()
        self.saved.addItem("Saved models…")
        if not df.empty:
            self.saved.addItems(df["name"].tolist())
        self.saved.blockSignals(False)

    def run(self) -> None:
        errs = validate(self.canvas.pipeline)
        if errs:
            QMessageBox.warning(self, "Model", "\n".join(errs))
            return
        if self.app.risk_service.active() is None:
            QMessageBox.information(self, "No risk model", "Fit a risk model first (Risk model tab).")
            return
        self.canvas.pipeline.name = self.name.text().strip() or "Untitled TLH model"
        self.run_btn.setEnabled(False)
        self.log.clear()
        self.app.status("Running TLH model…")
        run_task(self.svc.run, self.canvas.pipeline, self.ctx.current_entity_id, on_done=self._done, on_error=self._fail,
                 on_progress=lambda m: (self.log.appendPlainText(m), self.app.status(m)))

    def _done(self, res) -> None:
        self.run_btn.setEnabled(True)
        self.log.setPlainText("\n".join(res.log))
        self.k_uni.set(str(res.universe_size), "")
        self.k_n.set(str(int((res.weights > 0).sum())), res.basket_name or "not saved")
        d = res.diagnostics
        te = d.get("tracking_error") if isinstance(d, dict) else None
        if te is None and isinstance(d, dict) and isinstance(d.get("transition"), dict):
            te = d["transition"].get("te_to_target")
        self.k_te.set(pct(te), f"vs {res.benchmark}")
        hs = res.harvest_summary
        self.k_run.set(f"#{res.harvest_run_id}" if res.harvest_run_id else "—",
                       f"loss ${hs.get('harvested_loss', 0):,.0f} · TE {hs.get('te_before', 0):.2%}→{hs.get('te_after', 0):.2%}" if hs else "no Harvest block")
        self.weights.set_frame(res.weights.rename("weight").reset_index().rename(columns={"index": "symbol"}))
        self.tabs.setCurrentIndex(0 if not res.ok else 1)
        self.app.status("TLH model finished." if res.ok else "TLH model failed; see run log.")
        if res.ok:
            self.data_changed.emit()

    def _fail(self, msg: str) -> None:
        self.run_btn.setEnabled(True)
        self.log.appendPlainText(msg)
        self.app.error(msg)
