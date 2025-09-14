import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from prep.core import build_prompt, fetch_jd_text, read_cv_text
from prep.llm import call_openai
from prep.render import render_markdown

APP_TITLE = "Interview Prep Assistant (JD + CV → JSON + Markdown)"


# ---------------------- Main Window ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1100, 750)

        # Top form
        form_box = QGroupBox("Inputs")
        form = QFormLayout()

        self.jd_url = QLineEdit()
        self.jd_text = QTextEdit()
        self.jd_text.setPlaceholderText(
            "Paste JD text here (leave empty if using URL)."
        )

        self.cv_path_edit = QLineEdit()
        self.cv_path_edit.setReadOnly(True)
        pick_cv_btn = QPushButton("Choose CV…")
        pick_cv_btn.clicked.connect(self.pick_cv)

        cv_row = QWidget()
        cv_h = QHBoxLayout(cv_row)
        cv_h.setContentsMargins(0, 0, 0, 0)
        cv_h.addWidget(self.cv_path_edit)
        cv_h.addWidget(pick_cv_btn)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-5.1-mini"]
        )
        self.model_combo.setCurrentText("gpt-4o-mini")

        self.out_base_edit = QLineEdit(str(Path.cwd() / "prep_brief"))
        pick_out_btn = QPushButton("Output Base…")
        pick_out_btn.clicked.connect(self.pick_out_base)
        out_row = QWidget()
        out_h = QHBoxLayout(out_row)
        out_h.setContentsMargins(0, 0, 0, 0)
        out_h.addWidget(self.out_base_edit)
        out_h.addWidget(pick_out_btn)

        form.addRow(QLabel("JD URL:"), self.jd_url)
        form.addRow(QLabel("JD Text:"), self.jd_text)
        form.addRow(QLabel("CV File:"), cv_row)
        form.addRow(QLabel("Model:"), self.model_combo)
        form.addRow(QLabel("Output Base (without extension):"), out_row)

        form_box.setLayout(form)

        # Controls
        run_btn = QPushButton("Generate")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_prep)
        self.run_btn = run_btn

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)

        controls = QWidget()
        ch = QHBoxLayout(controls)
        ch.addWidget(run_btn)
        ch.addWidget(self.progress, 1)
        ch.addWidget(self.status_label, 3)

        # Markdown preview
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Markdown preview will appear here.")
        # setMarkdown is supported; if you prefer plain, switch to setPlainText
        self.preview.setAcceptRichText(False)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(form_box)
        layout.addWidget(controls)
        layout.addWidget(QLabel("Markdown Preview:"))
        layout.addWidget(self.preview, 1)
        self.setCentralWidget(central)

        # Load .env and check key
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            QMessageBox.warning(
                self,
                "OpenAI API Key",
                "OPENAI_API_KEY is not set. Add it to a .env file or environment.",
            )

        # Worker placeholder
        self.worker: Optional[PrepWorker] = None

    def pick_cv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CV (PDF or TXT)",
            str(Path.home()),
            "CV Files (*.pdf *.txt);;All Files (*)",
        )
        if path:
            self.cv_path_edit.setText(path)

    def pick_out_base(self):
        # Choose directory, keep filename stem
        base = Path(self.out_base_edit.text() or (Path.cwd() / "prep_brief"))
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(base.parent)
        )
        if directory:
            self.out_base_edit.setText(str(Path(directory) / base.name))

    def run_prep(self):
        # Validate mutually exclusive JD inputs
        url = self.jd_url.text().strip()
        jdt = self.jd_text.toPlainText().strip()
        if not url and not jdt:
            QMessageBox.warning(
                self, "Input needed", "Provide either a JD URL or paste JD text."
            )
            return
        if url and jdt:
            QMessageBox.warning(self, "Pick one", "Use JD URL OR JD text (not both).")
            return

        cvp = self.cv_path_edit.text().strip()
        if not cvp:
            QMessageBox.warning(
                self, "CV required", "Please choose your CV file (PDF/TXT)."
            )
            return

        model = self.model_combo.currentText().strip()
        out_base = Path(
            self.out_base_edit.text().strip() or (Path.cwd() / "prep_brief")
        )

        # UI state
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Running…")

        # Start worker
        self.worker = PrepWorker(url, jdt, cvp, model, out_base)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, result_json: dict, md_text: str, base_path: str):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.preview.setPlainText(md_text)
        self.status_label.setText(f"Done. Saved:\n{base_path}.json\n{base_path}.md")
        # Optional: validate required keys for peace of mind
        required = {
            "role_summary",
            "top_required_skills",
            "strong_overlaps",
            "gaps_and_risks",
            "likely_tech_questions",
            "behavioral_questions",
            "talking_points",
            "quick_upskilling_plan",
        }
        missing = required - set(result_json)
        if missing:
            QMessageBox.information(
                self, "Note", f"Missing keys in output: {sorted(missing)}"
            )

    def on_failed(self, message: str):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText("Failed.")
        QMessageBox.critical(self, "Error", message)


# ---------------------- Worker Thread ----------------------
class PrepWorker(QThread):
    done = Signal(dict, str, str)  # result_json, md_text, base_path
    failed = Signal(str)

    def __init__(
        self, jd_url: str, jd_text: str, cv_path: str, model: str, out_base: Path
    ):
        super().__init__()
        self.jd_url = jd_url.strip()
        self.jd_text = jd_text.strip()
        self.cv_path = Path(cv_path) if cv_path else None
        self.model = model.strip()
        self.out_base = out_base

    def run(self):
        try:
            if not self.cv_path:
                raise ValueError("Select a CV file.")
            jd = fetch_jd_text(self.jd_url or None, self.jd_text or None)
            cv = read_cv_text(self.cv_path)
            prompt = build_prompt(jd, cv)
            result = call_openai(prompt, model=self.model)
            md = render_markdown(result)

            # Save files
            self.out_base.parent.mkdir(parents=True, exist_ok=True)
            json_path = self.out_base.with_suffix(".json")
            md_path = self.out_base.with_suffix(".md")

            json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            md_path.write_text(md, encoding="utf-8")

            self.done.emit(result, md, str(self.out_base))
        except Exception as e:
            # Bubble message with traceback-ish detail
            self.failed.emit(f"{type(e).__name__}: {e}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
