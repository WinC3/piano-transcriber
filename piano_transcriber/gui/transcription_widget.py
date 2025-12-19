"""
Main transcription widget with drag-and-drop, progress tracking, and settings.
"""

import os
from pathlib import Path
from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QListWidget, QListWidgetItem,
    QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit, QSplitter,
    QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QPalette

from ..inference import PianoTranscriber, get_latest_checkpoint


class TranscriptionWorker(QThread):
    """Worker thread for non-blocking transcription."""
    
    progress_updated = pyqtSignal(int)  # Progress percentage
    status_updated = pyqtSignal(str)    # Status message
    file_completed = pyqtSignal(str, str, bool)  # file_path, output_path, success
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str, str)  # title, message
    
    def __init__(self, files: List[str], output_dir: str, settings: dict):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.settings = settings
        self.should_stop = False
        self.transcriber = None
    
    def run(self):
        """Run transcription in background thread."""
        try:
            # Load model
            self.status_updated.emit("Loading model...")
            model_path = self.settings.get('model_path') or get_latest_checkpoint()
            
            if not model_path:
                self.error_occurred.emit("Model Error", "No trained model found. Please train a model first.")
                return
            
            self.transcriber = PianoTranscriber(model_path)
            
            total_files = len(self.files)
            for i, file_path in enumerate(self.files):
                if self.should_stop:
                    break
                
                try:
                    file_name = Path(file_path).stem
                    self.status_updated.emit(f"Processing {file_name}...")
                    
                    # Determine output path
                    output_format = self.settings.get('output_format', 'midi')
                    ext = '.mid' if output_format == 'midi' else '.json'
                    output_path = Path(self.output_dir) / f"{file_name}{ext}"
                    
                    # Transcribe
                    predictions = self.transcriber.transcribe_audio(
                        file_path,
                        onset_threshold=self.settings.get('onset_threshold', 0.5),
                        frame_threshold=self.settings.get('frame_threshold', 0.5)
                    )
                    
                    # Save output
                    if output_format == 'midi':
                        self.transcriber.predictions_to_midi(predictions, output_path)
                    else:
                        import json
                        notes = self.transcriber.predictions_to_json(predictions)
                        with open(output_path, 'w') as f:
                            json.dump(notes, f, indent=2)
                    
                    self.file_completed.emit(file_path, str(output_path), True)
                    
                except Exception as e:
                    self.file_completed.emit(file_path, "", False)
                    self.error_occurred.emit("Transcription Error", f"Error processing {Path(file_path).name}: {str(e)}")
                
                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)
            
            if not self.should_stop:
                self.status_updated.emit("Transcription completed!")
            
        except Exception as e:
            self.error_occurred.emit("Error", f"Transcription failed: {str(e)}")
        
        finally:
            self.finished.emit()
    
    def stop(self):
        """Stop transcription."""
        self.should_stop = True


class DropArea(QListWidget):
    """Drag and drop area for audio files."""
    
    files_dropped = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        
        # Styling
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            QListWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f9f9f9;
                color: #333;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
                color: #333;
                background-color: transparent;
            }
            QListWidget::item:selected {
                background-color: #d4eaff;
                color: #000;
            }
        """)
        
        # Add placeholder text
        self.add_placeholder_item()
    
    def add_placeholder_item(self):
        """Add placeholder text when empty."""
        if self.count() == 0:
            item = QListWidgetItem("Drag audio files here or click 'Add Files' button")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setForeground(QPalette().color(QPalette.ColorRole.PlaceholderText))
            self.addItem(item)
    
    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Handle drag move event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle file drop event."""
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                # Check if it's an audio file
                if any(file_path.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']):
                    files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
    
    def add_files(self, files: List[str]):
        """Add files to the list."""
        # Remove placeholder if present
        if self.count() > 0:
            first_item = self.item(0)
            if first_item.flags() == Qt.ItemFlag.NoItemFlags:
                self.clear()
        
        for file_path in files:
            item = QListWidgetItem(Path(file_path).name)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.addItem(item)
    
    def clear_files(self):
        """Clear all files."""
        self.clear()
        self.add_placeholder_item()
    
    def get_files(self) -> List[str]:
        """Get list of file paths."""
        files = []
        for i in range(self.count()):
            item = self.item(i)
            if item.flags() != Qt.ItemFlag.NoItemFlags:  # Skip placeholder
                file_path = item.data(Qt.ItemDataRole.UserRole)
                if file_path:
                    files.append(file_path)
        return files


class TranscriptionWidget(QWidget):
    """Main transcription widget."""
    
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.transcription_worker = None
        self.transcription_in_progress = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # File selection area
        file_group = QGroupBox("Audio Files")
        file_layout = QVBoxLayout()
        
        # Drag and drop area
        self.file_list = DropArea()
        self.file_list.files_dropped.connect(self.add_files)
        file_layout.addWidget(self.file_list)
        
        # File buttons
        file_button_layout = QHBoxLayout()
        
        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self.select_files)
        file_button_layout.addWidget(self.add_files_btn)
        
        self.clear_files_btn = QPushButton("Clear")
        self.clear_files_btn.clicked.connect(self.clear_files)
        file_button_layout.addWidget(self.clear_files_btn)
        
        file_button_layout.addStretch()
        file_layout.addLayout(file_button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Settings area
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        
        # Output format
        settings_layout.addWidget(QLabel("Output Format:"), 0, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["MIDI", "JSON"])
        settings_layout.addWidget(self.output_format_combo, 0, 1)
        
        # Onset threshold
        settings_layout.addWidget(QLabel("Onset Threshold:"), 1, 0)
        self.onset_threshold_spin = QDoubleSpinBox()
        self.onset_threshold_spin.setRange(0.0, 1.0)
        self.onset_threshold_spin.setSingleStep(0.1)
        self.onset_threshold_spin.setValue(0.5)
        self.onset_threshold_spin.setDecimals(2)
        settings_layout.addWidget(self.onset_threshold_spin, 1, 1)
        
        # Frame threshold
        settings_layout.addWidget(QLabel("Frame Threshold:"), 2, 0)
        self.frame_threshold_spin = QDoubleSpinBox()
        self.frame_threshold_spin.setRange(0.0, 1.0)
        self.frame_threshold_spin.setSingleStep(0.1)
        self.frame_threshold_spin.setValue(0.5)
        self.frame_threshold_spin.setDecimals(2)
        settings_layout.addWidget(self.frame_threshold_spin, 2, 1)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Progress area
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.output_dir_btn = QPushButton("Select Output Directory...")
        self.output_dir_btn.clicked.connect(self.select_output_directory)
        button_layout.addWidget(self.output_dir_btn)
        
        button_layout.addStretch()
        
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.transcribe_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_transcription)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Default output directory
        self.output_directory = str(Path.home() / "Desktop")
    
    def add_files(self, files: List[str]):
        """Add files to the file list."""
        self.file_list.add_files(files)
        self.status_message.emit(f"Added {len(files)} file(s)")
    
    def select_files(self):
        """Open file dialog to select audio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)"
        )
        
        if files:
            self.add_files(files)
    
    def clear_files(self):
        """Clear all files."""
        self.file_list.clear_files()
        self.status_message.emit("File list cleared")
    
    def select_output_directory(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_directory
        )
        
        if directory:
            self.output_directory = directory
            self.status_message.emit(f"Output directory: {directory}")
    
    def start_transcription(self):
        """Start transcription process."""
        files = self.file_list.get_files()
        
        if not files:
            QMessageBox.warning(self, "No Files", "Please add some audio files first.")
            return
        
        # Prepare settings
        settings = {
            'output_format': self.output_format_combo.currentText().lower(),
            'onset_threshold': self.onset_threshold_spin.value(),
            'frame_threshold': self.frame_threshold_spin.value(),
        }
        
        # Start transcription worker
        self.transcription_worker = TranscriptionWorker(files, self.output_directory, settings)
        
        # Connect signals
        self.transcription_worker.progress_updated.connect(self.progress_bar.setValue)
        self.transcription_worker.status_updated.connect(self.status_label.setText)
        self.transcription_worker.file_completed.connect(self.on_file_completed)
        self.transcription_worker.error_occurred.connect(self.error_occurred)
        self.transcription_worker.finished.connect(self.on_transcription_finished)
        
        # Update UI state
        self.transcription_in_progress = True
        self.transcribe_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Start worker
        self.transcription_worker.start()
        
        self.status_message.emit("Transcription started")
    
    def stop_transcription(self):
        """Stop transcription process."""
        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.stop()
            self.transcription_worker.wait(3000)  # Wait up to 3 seconds
            
            if self.transcription_worker.isRunning():
                self.transcription_worker.terminate()
                self.transcription_worker.wait()
            
            self.status_label.setText("Transcription stopped")
            self.status_message.emit("Transcription stopped by user")
        
        self.on_transcription_finished()
    
    @pyqtSlot(str, str, bool)
    def on_file_completed(self, file_path: str, output_path: str, success: bool):
        """Handle file completion."""
        file_name = Path(file_path).name
        if success:
            self.status_message.emit(f"Completed: {file_name}")
        else:
            self.status_message.emit(f"Failed: {file_name}")
    
    @pyqtSlot()
    def on_transcription_finished(self):
        """Handle transcription completion."""
        self.transcription_in_progress = False
        self.transcribe_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)