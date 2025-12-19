"""
Main window for Piano Transcriber desktop application.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QMenuBar, QStatusBar, 
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from .transcription_widget import TranscriptionWidget


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Transcriber")
        self.setMinimumSize(600, 500)
        self.resize(800, 600)
        
        # Initialize UI
        self.init_ui()
        
        # Status bar for showing messages
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create menu bar
        self.create_menu_bar()
        
        # Create central widget
        self.transcription_widget = TranscriptionWidget()
        self.setCentralWidget(self.transcription_widget)
        
        # Connect signals
        self.transcription_widget.status_message.connect(self.show_status_message)
        self.transcription_widget.error_occurred.connect(self.show_error_message)
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Open files action
        open_action = QAction('&Open Audio File(s)...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        # About action
        about_action = QAction('&About...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def open_files(self):
        """Open file dialog to select audio files."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg)")
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.transcription_widget.add_files(files)
    
    def show_status_message(self, message: str, timeout: int = 5000):
        """Show message in status bar."""
        self.status_bar.showMessage(message, timeout)
    
    def show_error_message(self, title: str, message: str):
        """Show error dialog."""
        QMessageBox.critical(self, title, message)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, 
            "About Piano Transcriber",
            "<h3>Piano Transcriber</h3>"
            "<p>Neural network-based automatic piano transcription using the "
            "\"Onsets and Frames\" model.</p>"
            "<p>Built with PyTorch and PyQt6.</p>"
            "<p><a href='https://github.com/winc3/project-one'>GitHub Repository</a></p>"
        )
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Check if transcription is in progress
        if hasattr(self.transcription_widget, 'transcription_in_progress') and \
           self.transcription_widget.transcription_in_progress:
            reply = QMessageBox.question(
                self, 
                'Transcription in Progress',
                'Transcription is currently running. Are you sure you want to quit?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Stop transcription if possible
                self.transcription_widget.stop_transcription()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()